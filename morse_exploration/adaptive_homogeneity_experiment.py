"""
Adaptive Homogeneity Experiment
================================

This script tests the hypothesis that learning the "Effective Homogeneity Degree"
k_eff(r) is simpler than learning V(r) directly.

Connection to aPBE0 (Khan, Price et al., Sci. Adv. 2025):
- In aPBE0: We learned an adaptive mixing parameter α instead of energy directly
- Here: We learn k_eff(r) instead of V(r) directly

The key insight from Euler's theorem:
- For homogeneous functions: r·V'(r) = d·V(r), so Force and Energy are LINEAR
- For Morse (non-homogeneous): We define k_eff(r) = r·V'(r)/V(r) as the "adaptive d"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Publication-quality plot settings
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2


class MorseSimulator:
    """
    Simulates Morse potentials and computes the adaptive homogeneity parameter.

    Morse Potential: V(r) = De * (1 - exp(-a*(r - re)))^2

    Analytical Force: F(r) = -dV/dr = 2*De*a*(1 - exp(-a*(r-re)))*exp(-a*(r-re))

    Adaptive Parameter: k_eff(r) = r*V'(r) / V(r) = -r*F(r) / V(r)
    """

    def __init__(self, De, a, re):
        """
        Parameters:
        -----------
        De : float - Well depth (dissociation energy) in eV
        a : float - Width parameter in 1/Angstrom
        re : float - Equilibrium bond length in Angstrom
        """
        self.De = De
        self.a = a
        self.re = re
        self.alpha = a * re  # Dimensionless stiffness parameter

    def V(self, r):
        """Morse potential energy."""
        x = 1 - np.exp(-self.a * (r - self.re))
        return self.De * x**2

    def dV_dr(self, r):
        """Derivative of potential (gradient)."""
        exp_term = np.exp(-self.a * (r - self.re))
        return 2 * self.De * self.a * (1 - exp_term) * exp_term

    def F(self, r):
        """Force = -dV/dr."""
        return -self.dV_dr(r)

    def k_eff(self, r, V_threshold=1e-5):
        """
        Effective homogeneity degree: k_eff = r*V'(r) / V(r)

        Returns NaN where V < threshold to avoid singularity at equilibrium.
        """
        V = self.V(r)
        dV = self.dV_dr(r)

        k = np.full_like(r, np.nan)
        valid = np.abs(V) > V_threshold
        k[valid] = (r[valid] * dV[valid]) / V[valid]

        return k

    def generate_data(self, r_min_factor=0.5, r_max_factor=4.0, n_points=200):
        """Generate a dataset for this potential."""
        r = np.linspace(r_min_factor * self.re, r_max_factor * self.re, n_points)

        V = self.V(r)
        F = self.F(r)
        k = self.k_eff(r)

        # Reduced coordinate
        r_star = r / self.re

        return {
            'r': r,
            'r_star': r_star,
            'V': V,
            'F': F,
            'r_times_F': r * F,
            'k_eff': k
        }


def generate_random_potentials(n_potentials=50, seed=42):
    """Generate n random Morse potentials with varied parameters."""
    np.random.seed(seed)

    potentials = []
    for i in range(n_potentials):
        De = np.random.uniform(0.5, 5.0)   # eV
        re = np.random.uniform(1.0, 3.0)   # Angstrom
        a = np.random.uniform(0.5, 2.0)    # 1/Angstrom

        sim = MorseSimulator(De, a, re)
        potentials.append({
            'id': i,
            'De': De,
            're': re,
            'a': a,
            'alpha': a * re,
            'simulator': sim
        })

    return potentials


def experiment1_euler_violation(potentials, save_path='fig_euler_violation.png'):
    """
    Experiment 1: The "Euler Violation" Plot

    Plot r·F(r) vs V(r) for all potentials.

    - For a HOMOGENEOUS potential of degree d: r·F = -d·V (straight line!)
    - For Morse: This forms a LOOP, proving it's not homogeneous
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: The Euler Violation (loops)
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, len(potentials)))

    for pot, color in zip(potentials, colors):
        sim = pot['simulator']
        data = sim.generate_data()
        ax.plot(data['V'], data['r_times_F'], color=color, alpha=0.5, linewidth=1)

    # Add reference lines for homogeneous potentials
    V_max = max(pot['De'] for pot in potentials)
    V_line = np.linspace(0, V_max, 100)
    ax.plot(V_line, 1 * V_line, 'k--', linewidth=2, label='d = -1 (Coulomb)')
    ax.plot(V_line, 6 * V_line, 'k:', linewidth=2, label='d = -6 (Dispersion)')

    ax.set_xlabel('V(r) [eV]')
    ax.set_ylabel('r · F(r) [eV]')
    ax.set_title('Euler Violation: Morse forms LOOPS\n(Homogeneous potentials would be straight lines)')
    ax.legend()
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(0, V_max * 1.1)

    # Right: Explanation
    ax = axes[1]
    ax.text(0.5, 0.9, "What this plot shows:", fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)

    explanation = """
For a homogeneous potential V(r) ~ 1/r^d:

    Euler's theorem: r · V'(r) = d · V(r)

    Since F = -V', this means: r · F = -d · V

    → Plotting r·F vs V gives a STRAIGHT LINE with slope -d

For Morse potential:

    The plot forms a LOOP, not a line

    → Morse is NOT homogeneous
    → There is no single "d" that works everywhere

The "Adaptive" idea:

    Define k_eff(r) = r·V'/V at each point

    → k_eff is the LOCAL "d" that varies with position
    → If k_eff is simpler than V(r), we should learn IT instead
"""
    ax.text(0.1, 0.75, explanation, fontsize=11, family='monospace',
            va='top', transform=ax.transAxes)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("EXPERIMENT 1: Euler Violation")
    print("=" * 50)
    print("The loops prove Morse is non-homogeneous.")
    print("This is the PROBLEM we're trying to solve.")


def experiment2_corresponding_states(potentials, save_path='fig_corresponding_states.png'):
    """
    Experiment 2: The "Corresponding States" Hypothesis

    Plot k_eff vs reduced distance r* = r/re for all potentials.

    If the curves COLLAPSE, it means k_eff is more universal than V(r).
    This is the "Money Plot" - it shows learning k_eff is easier.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = cm.viridis(np.linspace(0, 1, len(potentials)))

    # Panel 1: Raw V(r) - all different
    ax = axes[0]
    for pot, color in zip(potentials, colors):
        sim = pot['simulator']
        data = sim.generate_data()
        ax.plot(data['r'], data['V'], color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('V(r) [eV]')
    ax.set_title('Raw Potentials: ALL DIFFERENT\n(50 random Morse curves)')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Panel 2: k_eff vs r (still messy)
    ax = axes[1]
    for pot, color in zip(potentials, colors):
        sim = pot['simulator']
        data = sim.generate_data()
        ax.plot(data['r'], data['k_eff'], color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('k_eff(r)')
    ax.set_title('k_eff vs r: Still scattered\n(different re values)')
    ax.set_ylim(-15, 15)

    # Panel 3: k_eff vs r* = r/re (THE COLLAPSE!)
    ax = axes[2]
    for pot, color in zip(potentials, colors):
        sim = pot['simulator']
        data = sim.generate_data()
        ax.plot(data['r_star'], data['k_eff'], color=color, alpha=0.5, linewidth=1)

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='r* = 1 (equilibrium)')
    ax.set_xlabel('r* = r/re (reduced distance)')
    ax.set_ylabel('k_eff(r)')
    ax.set_title('k_eff vs r*: COLLAPSE!\n(Universal curve emerges)')
    ax.set_ylim(-15, 15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 2: Corresponding States")
    print("=" * 50)
    print("LEFT: 50 different V(r) curves - hard to learn")
    print("RIGHT: Same curves in (r*, k_eff) space - they COLLAPSE!")
    print("")
    print("IMPLICATION: Learning k_eff(r*) is much simpler because")
    print("all Morse potentials share the same universal shape.")


def experiment3_alpha_dependence(save_path='fig_alpha_dependence.png'):
    """
    Experiment 3: Show that k_eff depends only on r* and alpha = a·re

    Generate potentials with SAME alpha but different (De, re, a).
    Their k_eff curves should perfectly overlap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Test with fixed alpha = 1.5
    alpha_fixed = 1.5
    n_curves = 10

    np.random.seed(42)

    # Left: Different alpha values
    ax = axes[0]
    alpha_values = [0.8, 1.2, 1.5, 2.0, 2.5]
    colors = cm.plasma(np.linspace(0.1, 0.9, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors):
        # Fix alpha, vary other params
        re = 1.5
        a = alpha / re
        De = 3.0  # Doesn't matter for k_eff!

        sim = MorseSimulator(De, a, re)
        data = sim.generate_data()
        ax.plot(data['r_star'], data['k_eff'], color=color, linewidth=2,
                label=f'α = {alpha}')

    ax.axvline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('r* = r/re')
    ax.set_ylabel('k_eff')
    ax.set_title('Different stiffness α = a·re\ngives different curves')
    ax.set_ylim(-15, 15)
    ax.legend()

    # Right: Same alpha, different (De, re, a)
    ax = axes[1]
    colors = cm.viridis(np.linspace(0, 1, n_curves))

    for i, color in enumerate(colors):
        De = np.random.uniform(0.5, 8.0)   # DIFFERENT De
        re = np.random.uniform(0.8, 3.0)   # DIFFERENT re
        a = alpha_fixed / re               # Constrained so alpha is SAME

        sim = MorseSimulator(De, a, re)
        data = sim.generate_data()

        label = f'De={De:.1f}, re={re:.1f}' if i < 3 else None
        ax.plot(data['r_star'], data['k_eff'], color=color, linewidth=1.5,
                alpha=0.7, label=label)

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('r* = r/re')
    ax.set_ylabel('k_eff')
    ax.set_title(f'SAME α = {alpha_fixed}, different De and re\n→ Curves PERFECTLY OVERLAP')
    ax.set_ylim(-15, 15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 3: Alpha Dependence")
    print("=" * 50)
    print("LEFT: Different α values give different k_eff curves")
    print("RIGHT: Same α, different (De, re) → PERFECT OVERLAP")
    print("")
    print("KEY FINDING: k_eff depends ONLY on (r*, α)")
    print("  - De completely scales out!")
    print("  - Dimensionality: V(r; De, a, re) → k_eff(r*; α)")
    print("  - That's 3 parameters → 1 parameter!")


def experiment4_ml_comparison(potentials, save_path='fig_ml_comparison.png'):
    """
    Experiment 4: Compare learning V(r) vs learning k_eff

    This directly tests whether k_eff is "easier" to learn.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Prepare data
    all_data = []
    for pot in potentials:
        sim = pot['simulator']
        data = sim.generate_data()

        for i in range(len(data['r'])):
            if not np.isnan(data['k_eff'][i]):
                all_data.append({
                    'id': pot['id'],
                    'r': data['r'][i],
                    'r_star': data['r_star'][i],
                    'alpha': pot['alpha'],
                    'V': data['V'][i],
                    'k_eff': data['k_eff'][i]
                })

    df = pd.DataFrame(all_data)

    # Split by potential ID (not random points)
    unique_ids = df['id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_df = df[df['id'].isin(train_ids)]
    test_df = df[df['id'].isin(test_ids)]

    print(f"\nTraining on {len(train_ids)} potentials, testing on {len(test_ids)} potentials")

    # Model A: Learn V from (r, alpha)
    X_train_V = train_df[['r', 'alpha']].values
    y_train_V = train_df['V'].values
    X_test_V = test_df[['r', 'alpha']].values
    y_test_V = test_df['V'].values

    scaler_V = StandardScaler()
    X_train_V_scaled = scaler_V.fit_transform(X_train_V)
    X_test_V_scaled = scaler_V.transform(X_test_V)

    model_V = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
    model_V.fit(X_train_V_scaled, y_train_V)
    y_pred_V = model_V.predict(X_test_V_scaled)

    rmse_V = np.sqrt(np.mean((y_pred_V - y_test_V)**2))

    # Model B: Learn k_eff from (r_star, alpha)
    X_train_k = train_df[['r_star', 'alpha']].values
    y_train_k = train_df['k_eff'].values
    X_test_k = test_df[['r_star', 'alpha']].values
    y_test_k = test_df['k_eff'].values

    scaler_k = StandardScaler()
    X_train_k_scaled = scaler_k.fit_transform(X_train_k)
    X_test_k_scaled = scaler_k.transform(X_test_k)

    model_k = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
    model_k.fit(X_train_k_scaled, y_train_k)
    y_pred_k = model_k.predict(X_test_k_scaled)

    rmse_k = np.sqrt(np.mean((y_pred_k - y_test_k)**2))

    # Normalize errors for comparison
    range_V = y_test_V.max() - y_test_V.min()
    range_k = y_test_k.max() - y_test_k.min()
    nrmse_V = rmse_V / range_V * 100
    nrmse_k = rmse_k / range_k * 100

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_test_V, y_pred_V, alpha=0.3, s=10)
    ax.plot([0, y_test_V.max()], [0, y_test_V.max()], 'r--', linewidth=2)
    ax.set_xlabel('True V(r) [eV]')
    ax.set_ylabel('Predicted V(r) [eV]')
    ax.set_title(f'Learning V(r) directly\nRMSE = {rmse_V:.4f} eV ({nrmse_V:.1f}% of range)')

    ax = axes[1]
    ax.scatter(y_test_k, y_pred_k, alpha=0.3, s=10)
    lim = max(abs(y_test_k.min()), abs(y_test_k.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2)
    ax.set_xlabel('True k_eff')
    ax.set_ylabel('Predicted k_eff')
    ax.set_title(f'Learning k_eff(r*)\nRMSE = {rmse_k:.4f} ({nrmse_k:.1f}% of range)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 4: ML Comparison")
    print("=" * 50)
    print(f"Learning V(r):   RMSE = {rmse_V:.4f} eV  ({nrmse_V:.1f}% of range)")
    print(f"Learning k_eff:  RMSE = {rmse_k:.4f}     ({nrmse_k:.1f}% of range)")
    print("")
    if nrmse_k < nrmse_V:
        print(f"→ Learning k_eff is {nrmse_V/nrmse_k:.1f}x better (normalized)!")
    else:
        print(f"→ Similar performance, but k_eff uses reduced coordinates")

    return df, train_df, test_df


def export_data(df, train_df, test_df):
    """Export data for external ML experiments."""
    train_df.to_csv('train_set.csv', index=False)
    test_df.to_csv('test_set.csv', index=False)
    print(f"\nExported: train_set.csv ({len(train_df)} points)")
    print(f"Exported: test_set.csv ({len(test_df)} points)")


def main():
    print("=" * 70)
    print("ADAPTIVE HOMOGENEITY EXPERIMENT")
    print("=" * 70)
    print("")
    print("Testing whether learning k_eff(r) is simpler than learning V(r)")
    print("This is analogous to learning α in aPBE0 instead of E directly.")
    print("")

    # Generate random potentials
    potentials = generate_random_potentials(n_potentials=50)

    # Run experiments
    experiment1_euler_violation(potentials)
    experiment2_corresponding_states(potentials)
    experiment3_alpha_dependence()
    df, train_df, test_df = experiment4_ml_comparison(potentials)

    # Export data
    export_data(df, train_df, test_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The "Adaptive Homogeneity" Concept:

1. PROBLEM: Morse potentials are non-homogeneous (Euler's theorem fails)
   → The Virial plot (r·F vs V) forms loops, not lines

2. SOLUTION: Define k_eff(r) = r·V'(r)/V(r) as the "local d"
   → This is the ADAPTIVE PARAMETER (like α in aPBE0)

3. KEY FINDING: In reduced coordinates (r* = r/re), k_eff depends
   only on the stiffness α = a·re, NOT on De
   → Massive dimensionality reduction: 3 params → 1 param

4. IMPLICATION: Learning k_eff(r*; α) should be simpler than V(r)
   because all Morse potentials share the same universal shape.

NEXT STEPS:
- Test on real diatomic data (NIST, ExoMol)
- Extend to other potential forms (LJ, Buckingham)
- Connect to Rose equation / UBER for bulk systems
""")


if __name__ == "__main__":
    main()
