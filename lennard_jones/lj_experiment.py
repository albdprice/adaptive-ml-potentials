"""
Lennard-Jones Experiment: Adaptive Universal Scaling
=====================================================

Demonstrates that:
1. LJ potentials with varying (epsilon, sigma) produce diverse binding curves
2. All curves collapse to a universal shape when scaled by (epsilon, sigma)
3. The LJ parameters vary SMOOTHLY (making them easy to ML)
4. Adaptive approach (learn epsilon, sigma) outperforms direct learning on extrapolation
5. Same principle as aPBE0 and Rose/UBER

Lennard-Jones potential:
    V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]

Parameters:
    epsilon: well depth (energy scale)
    sigma:   zero-crossing distance (length scale)

Reduced form:
    V*(r*) = 4 * [(1/r*)^12 - (1/r*)^6]   where r* = r/sigma, V* = V/epsilon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14


# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

def lennard_jones(r, epsilon, sigma):
    """LJ potential: V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]"""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def lj_force(r, epsilon, sigma):
    """LJ force: F(r) = -dV/dr = 4*epsilon * [12*sigma^12/r^13 - 6*sigma^6/r^7]"""
    return 4 * epsilon * (12 * sigma**12 / r**13 - 6 * sigma**6 / r**7)


def lj_reduced(r_star):
    """Reduced LJ: V*(r*) = 4 * [(1/r*)^12 - (1/r*)^6]"""
    return 4 * (r_star**(-12) - r_star**(-6))


# =============================================================================
# EXPERIMENT 1: LJ Potentials with Varying Parameters
# =============================================================================

def experiment1_lj_potentials(n_potentials=50, seed=42):
    """Show the diversity of LJ potentials with different (epsilon, sigma)."""
    np.random.seed(seed)

    epsilons = np.random.uniform(0.01, 0.5, n_potentials)  # eV
    sigmas = np.random.uniform(2.0, 4.0, n_potentials)      # Angstrom

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = cm.viridis(np.linspace(0, 1, n_potentials))

    # Panel 1: Raw LJ curves
    ax = axes[0]
    for i in range(n_potentials):
        r = np.linspace(0.9 * sigmas[i], 3.0 * sigmas[i], 100)
        V = lennard_jones(r, epsilons[i], sigmas[i])
        ax.plot(r, V, color=colors[i], alpha=0.5, linewidth=1)
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title(f'{n_potentials} LJ potentials\n(varying epsilon and sigma)')
    ax.set_ylim(-0.6, 1.0)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Distribution of epsilon
    ax = axes[1]
    ax.hist(epsilons, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('epsilon [eV]')
    ax.set_ylabel('Count')
    ax.set_title(f'Well depth distribution\nepsilon: {epsilons.min():.3f} - {epsilons.max():.3f} eV')
    ax.grid(True, alpha=0.3)

    # Panel 3: Distribution of sigma
    ax = axes[2]
    ax.hist(sigmas, bins=15, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('sigma [A]')
    ax.set_ylabel('Count')
    ax.set_title(f'Size parameter distribution\nsigma: {sigmas.min():.2f} - {sigmas.max():.2f} A')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig1_lj_potentials.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("EXPERIMENT 1: LJ Potentials with Varying Parameters")
    print("=" * 60)
    print(f"Generated {n_potentials} LJ potentials")
    print(f"  epsilon range: {epsilons.min():.3f} - {epsilons.max():.3f} eV")
    print(f"  sigma range:   {sigmas.min():.2f} - {sigmas.max():.2f} A")
    print("  Each potential has a different shape due to (epsilon, sigma)")

    return epsilons, sigmas


# =============================================================================
# EXPERIMENT 2: Universal Collapse
# =============================================================================

def experiment2_universal_collapse(epsilons, sigmas):
    """Show that all LJ curves collapse to ONE universal curve when scaled."""

    n_potentials = len(epsilons)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = cm.viridis(np.linspace(0, 1, n_potentials))

    # Panel 1: Raw curves
    ax = axes[0]
    for i in range(n_potentials):
        r = np.linspace(0.9 * sigmas[i], 3.0 * sigmas[i], 100)
        V = lennard_jones(r, epsilons[i], sigmas[i])
        ax.plot(r, V, color=colors[i], alpha=0.5, linewidth=1)
    ax.set_xlabel('r [A]')
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
    ax.set_xlabel('r [A]')
    ax.set_ylabel('V(r) / epsilon')
    ax.set_title('Scaled by epsilon only\n(still scattered)')
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
    ax.plot(r_theory, V_theory, 'k--', linewidth=3,
            label='Universal: 4[(1/r*)^12 - (1/r*)^6]')

    ax.set_xlabel('r* = r/sigma')
    ax.set_ylabel('V* = V/epsilon')
    ax.set_title('UNIVERSAL COLLAPSE!\n(all follow ONE curve)')
    ax.set_ylim(-1.5, 2.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(2**(1/6), color='red', linestyle=':', alpha=0.5,
               label=f'r*_min = 2^(1/6) = {2**(1/6):.3f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig2_lj_universal_collapse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nEXPERIMENT 2: Universal Collapse")
    print("=" * 60)
    print("LEFT:  50 different V(r) curves (all look different)")
    print("RIGHT: When scaled by (epsilon, sigma), ALL collapse to ONE curve!")
    print()
    print("IMPLICATION: All the 'chemistry' is encoded in just 2 parameters!")
    print("  epsilon = well depth (energy scale)")
    print("  sigma   = zero-crossing distance (length scale)")


# =============================================================================
# EXPERIMENT 3: Parameter Smoothness
# =============================================================================

def experiment3_parameter_smoothness():
    """Show that LJ parameters vary smoothly with descriptors."""

    n_points = 30

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scenario 1: Vary d1 (controls sigma)
    ax = axes[0]
    d1_range = np.linspace(0.5, 4.0, n_points)
    sigmas = 2.5 + 0.5 * d1_range
    epsilons = np.full(n_points, 0.1)  # fixed

    ax.plot(d1_range, sigmas, 'o-', label='sigma', linewidth=2, markersize=6, color='coral')
    ax.plot(d1_range, epsilons * 10, 's-', label='epsilon x10', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Descriptor d1 (atomic radius proxy)')
    ax.set_ylabel('Parameter value')
    ax.set_title('Varying d1 (size descriptor)\nsigma varies linearly, epsilon constant')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scenario 2: Vary d2 (controls epsilon)
    ax = axes[1]
    d2_range = np.linspace(0.5, 4.0, n_points)
    epsilons = 0.05 + 0.1 * d2_range
    sigmas = np.full(n_points, 3.0)  # fixed

    ax.plot(d2_range, epsilons, 'o-', label='epsilon', linewidth=2, markersize=6, color='steelblue')
    ax.plot(d2_range, sigmas, 's-', label='sigma', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Descriptor d2 (interaction strength proxy)')
    ax.set_ylabel('Parameter value')
    ax.set_title('Varying d2 (energy descriptor)\nepsilon varies linearly, sigma constant')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scenario 3: Both vary (realistic case)
    ax = axes[2]
    d1_range = np.linspace(0.5, 4.0, n_points)
    d2_range = np.linspace(0.5, 4.0, n_points)
    sigmas = 2.5 + 0.5 * d1_range
    epsilons = 0.05 + 0.1 * d2_range

    sc = ax.scatter(sigmas, epsilons, c=d1_range, cmap='viridis', s=80, edgecolors='black')
    ax.set_xlabel('sigma [A]')
    ax.set_ylabel('epsilon [eV]')
    ax.set_title('Parameter space\n(color = d1 value)')
    plt.colorbar(sc, ax=ax, label='d1')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig3_lj_parameter_smoothness.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nEXPERIMENT 3: Parameter Smoothness")
    print("=" * 60)
    print("LJ parameters vary SMOOTHLY with descriptors:")
    print("  - sigma ~ 2.5 + 0.5 * d1 (linear relationship)")
    print("  - epsilon ~ 0.05 + 0.1 * d2 (linear relationship)")
    print()
    print("THIS IS THE KEY RESULT!")
    print("Smooth variation means the parameters are EASY TO ML.")
    print("(Same argument that made alpha learnable in aPBE0)")


# =============================================================================
# EXPERIMENT 4: ML Comparison (The Actual Test!)
# =============================================================================

def experiment4_ml_comparison():
    """
    Compare ML approaches with focus on EXTRAPOLATION.

    Train on small atoms (low sigma), test on large atoms (high sigma).
    Uses a FIXED r-grid so that changing sigma genuinely changes the curve
    shape (nonlinear dependence), rather than sigma-scaled grids where
    sigma/r is constant and the problem becomes trivially linear.
    """
    np.random.seed(42)
    n_r_points = 50

    # Fixed r-grid covering range needed for all sigma values
    r_grid = np.linspace(2.5, 12.0, n_r_points)

    # --- Generate TRAINING data: small atoms ---
    train_data = {'X': [], 'params': [], 'curves': [], 'r_grid': r_grid}

    for i in range(80):
        d1 = np.random.uniform(0.5, 1.5)
        d2 = np.random.uniform(0.5, 1.5)

        sigma = 2.5 + 0.5 * d1    # Training: sigma ~ 2.75-3.25 A
        epsilon = 0.05 + 0.1 * d2  # epsilon ~ 0.10-0.20 eV

        V = lennard_jones(r_grid, epsilon, sigma)

        train_data['X'].append([d1, d2])
        train_data['params'].append([epsilon, sigma])
        train_data['curves'].append(V)

    # --- Generate TEST data: LARGE atoms (high sigma) - EXTRAPOLATION! ---
    test_data = {'X': [], 'params': [], 'curves': [], 'r_grid': r_grid}

    for i in range(40):
        d1 = np.random.uniform(2.5, 4.0)   # EXTRAPOLATION in d1
        d2 = np.random.uniform(0.5, 1.5)

        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2

        V = lennard_jones(r_grid, epsilon, sigma)

        test_data['X'].append([d1, d2])
        test_data['params'].append([epsilon, sigma])
        test_data['curves'].append(V)

    X_train = np.array(train_data['X'])
    Y_params_train = np.array(train_data['params'])
    Y_curves_train = np.array(train_data['curves'])
    X_test = np.array(test_data['X'])
    Y_params_test = np.array(test_data['params'])
    Y_curves_test = np.array(test_data['curves'])

    print("\nEXPERIMENT 4: ML Extrapolation Comparison")
    print("=" * 60)
    print(f"Training: {len(X_train)} small atoms (sigma ~ 2.75-3.25 A)")
    print(f"Testing:  {len(X_test)} LARGE atoms (sigma ~ 3.75-4.5 A) - EXTRAPOLATION!")

    # Scale data
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # --- Direct approach ---
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    Y_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())

    # --- Adaptive approach ---
    scaler_params = StandardScaler()
    Y_params_s = scaler_params.fit_transform(Y_params_train)
    model_params = Ridge(alpha=1.0)
    model_params.fit(X_train_s, Y_params_s)
    Y_params_pred = scaler_params.inverse_transform(model_params.predict(X_test_s))

    Y_pred_adaptive = []
    for i, (eps, sig) in enumerate(Y_params_pred):
        Y_pred_adaptive.append(lennard_jones(r_grid, max(eps, 0.001), max(sig, 0.1)))
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())

    print(f"\nDirect approach:   MSE = {mse_direct:.6f} eV^2, R^2 = {r2_direct:.4f}")
    print(f"Adaptive approach: MSE = {mse_adaptive:.6f} eV^2, R^2 = {r2_adaptive:.4f}")

    # --- Plot ---
    fig = plt.figure(figsize=(15, 10))
    axes = fig.subplots(2, 3)

    # Top row: Example predictions
    examples = [0, len(X_test) // 2, -1]
    for col, idx in enumerate(examples):
        ax = axes[0, col]
        r = r_grid
        eps, sig = test_data['params'][idx]

        ax.plot(r, Y_curves_test[idx], 'k-', linewidth=3, label='True')
        ax.plot(r, Y_pred_direct[idx], 'b--', linewidth=2, label='Direct ML')
        ax.plot(r, Y_pred_adaptive[idx], 'r:', linewidth=2, label='Adaptive ML')

        ax.set_xlabel('r [A]')
        ax.set_ylabel('V(r) [eV]')
        ax.set_title(f'Large atom sigma={sig:.2f} A\n(extrapolation from training)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        # Zoom into well region (repulsive wall washes out minima)
        ax.set_ylim(-0.3, 0.5)

    # Bottom left: MSE comparison
    ax = axes[1, 0]
    methods = ['Direct\n(learn V(r))', 'Adaptive\n(learn eps, sigma)']
    mses = [mse_direct, mse_adaptive]
    colors_bar = ['blue', 'red']
    ax.bar(methods, mses, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MSE [eV^2]')
    ax.set_title('Extrapolation Error')
    if mse_adaptive < mse_direct:
        improvement = (mse_direct - mse_adaptive) / mse_direct * 100
        ax.text(0.5, 0.7, f'Adaptive is\n{improvement:.0f}% better!',
                ha='center', fontsize=14, fontweight='bold', color='green',
                transform=ax.transAxes)

    # Bottom middle: R^2 comparison
    ax = axes[1, 1]
    r2s = [r2_direct, r2_adaptive]
    ax.bar(methods, r2s, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R^2 score')
    ax.set_title('Extrapolation R^2')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    # Bottom right: sigma extrapolation
    ax = axes[1, 2]
    ax.scatter(Y_params_test[:, 1], Y_params_pred[:, 1], alpha=0.7, s=60, c='tab:orange')
    lims = [2, max(Y_params_test[:, 1].max(), Y_params_pred[:, 1].max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect')

    train_sigma_max = Y_params_train[:, 1].max()
    ax.axvline(train_sigma_max, color='red', linestyle=':', linewidth=2,
               label=f'Training max sigma={train_sigma_max:.1f}')
    ax.axhline(train_sigma_max, color='red', linestyle=':', linewidth=2)

    ax.set_xlabel('True sigma [A]')
    ax.set_ylabel('Predicted sigma [A]')
    ax.set_title('sigma Extrapolation\n(predicting beyond training)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('fig4_lj_ml_comparison.png', dpi=150)
    plt.close(fig)

    print("\n" + "-" * 60)
    if mse_adaptive < mse_direct:
        print(f"RESULT: Adaptive is {(mse_direct - mse_adaptive) / mse_direct * 100:.0f}% better at extrapolation!")
    print("Same principle as Rose/UBER: physics equation + smooth parameters = better ML")
    print("-" * 60)

    return {
        'mse_direct': mse_direct, 'mse_adaptive': mse_adaptive,
        'r2_direct': r2_direct, 'r2_adaptive': r2_adaptive
    }


# =============================================================================
# EXPERIMENT 5: The aPBE0 Analogy
# =============================================================================

def experiment5_summary():
    """Create a summary figure showing the aPBE0 analogy for LJ."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: aPBE0 approach
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'aPBE0 Approach\n(Khan, Price et al. 2025)',
            fontsize=14, fontweight='bold', ha='center', va='top')

    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.88, 'Local Density Features', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.73, 'ML Model', ha='center', fontsize=11)

    ax.text(0.5, 0.55, 'alpha (exchange mixing)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.52),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.45, 'Physics Equation', ha='center', fontsize=11)

    ax.text(0.5, 0.35, 'E = (1-alpha)E_PBE + alpha*E_HF', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.annotate('', xy=(0.5, 0.18), xytext=(0.5, 0.32),
                arrowprops=dict(arrowstyle='->', lw=2))

    ax.text(0.5, 0.15, 'Accurate Energy', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral'))

    ax.text(0.5, 0.05, 'alpha is bounded [0,1], smooth, transferable',
            ha='center', fontsize=10, style='italic')
    ax.axis('off')

    # Right: LJ approach
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'Lennard-Jones Approach\n(This Work)',
            fontsize=14, fontweight='bold', ha='center', va='top')

    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.88, 'Local Atomic Environment', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.73, 'ML Model', ha='center', fontsize=11)

    ax.text(0.5, 0.55, '(epsilon, sigma)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.52),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.5, 0.45, 'Physics Equation', ha='center', fontsize=11)

    ax.text(0.5, 0.35, 'V = 4*eps*[(sig/r)^12 - (sig/r)^6]', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.annotate('', xy=(0.5, 0.18), xytext=(0.5, 0.32),
                arrowprops=dict(arrowstyle='->', lw=2))

    ax.text(0.5, 0.15, 'Accurate Energy', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral'))

    ax.text(0.5, 0.05, '(epsilon, sigma) are bounded, smooth, transferable',
            ha='center', fontsize=10, style='italic')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig5_lj_apbe0_analogy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nEXPERIMENT 5: The aPBE0 Analogy")
    print("=" * 60)
    print("Both approaches share the same philosophy:")
    print()
    print("  aPBE0:          descriptor -> alpha -> E via physics")
    print("  Rose/UBER:      descriptor -> (E_c, r_e, l) -> E via physics")
    print("  Lennard-Jones:  descriptor -> (epsilon, sigma) -> V via physics")
    print()
    print("The ML task becomes SIMPLER because:")
    print("  1. Parameters are bounded (physically constrained)")
    print("  2. Parameters vary smoothly with environment")
    print("  3. Physics equation enforces correct behavior")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("LENNARD-JONES EXPERIMENT: Adaptive Universal Scaling")
    print("=" * 70)
    print()
    print("Demonstrating that learning (epsilon, sigma) is simpler than V(r)")
    print("This is directly analogous to learning alpha in aPBE0")
    print()

    epsilons, sigmas = experiment1_lj_potentials()
    experiment2_universal_collapse(epsilons, sigmas)
    experiment3_parameter_smoothness()
    ml_results = experiment4_ml_comparison()
    experiment5_summary()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
KEY FINDINGS:

1. LJ DIVERSITY: Different (epsilon, sigma) produce diverse binding curves
   but ALL share the same universal shape

2. UNIVERSAL COLLAPSE: When scaled by (epsilon, sigma), all LJ curves
   collapse to ONE universal curve: V* = 4[(1/r*)^12 - (1/r*)^6]

3. SMOOTH PARAMETERS: epsilon and sigma vary SMOOTHLY with descriptors
   -> This is why they're easy to machine learn!

4. ML EXTRAPOLATION: Adaptive approach extrapolates better!
   - Train on small atoms (sigma ~ 2.75-3.25 A)
   - Test on LARGE atoms (sigma ~ 3.75-4.5 A)
   - Direct MSE = {ml_results['mse_direct']:.6f}
   - Adaptive MSE = {ml_results['mse_adaptive']:.6f}
   -> Physics equation provides correct shape even beyond training

5. aPBE0 ANALOGY: Same philosophy:
   - Don't learn V directly
   - Learn bounded, smooth parameters (epsilon, sigma)
   - Use physics equations for the final result

NEXT STEPS:
- Apply to real noble gas / van der Waals systems
- Test with proper atomic descriptors (SOAP, ACE)
- Compare with Rose/UBER results on same systems
""")


if __name__ == "__main__":
    main()
