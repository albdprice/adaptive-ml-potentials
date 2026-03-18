"""
Adaptive Parameter Learning for Interatomic Potentials
=======================================================

Demonstrating that learning adaptive parameters (E_c, r_e, l) is simpler
than learning the energy surface directly.

Connection to aPBE0 (Khan, Price et al., Sci. Adv. 2025):
---------------------------------------------------------
In aPBE0: Learn α (exchange mixing), then compute E = (1-α)E_PBE + α*E_HF
Here: Learn (E_c, r_e, l), then compute E from universal Rose curve

The Rose Equation (Universal Binding Energy Relation - UBER):
-------------------------------------------------------------
E(a*) = -E_c * (1 + a*) * exp(-a*)

where a* = (r - r_e) / l is a scaled/reduced distance

Parameters:
- E_c: Cohesive energy (well depth)
- r_e: Equilibrium distance
- l: Characteristic length (related to stiffness/bulk modulus)

Key insight: ALL binding curves collapse to ONE universal shape when
scaled by these three parameters. Learning (E_c, r_e, l) is simpler
than learning E(r) directly because these parameters are:
- Bounded and smooth
- Physically meaningful
- Transferable across systems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14


def rose_energy(r, E_c, r_e, l):
    """
    Universal Binding Energy Relation (Rose equation).

    E(r) = -E_c * (1 + a*) * exp(-a*)

    where a* = (r - r_e) / l

    At r = r_e: a* = 0, E = -E_c (minimum)
    As r -> inf: a* -> inf, E -> 0 (dissociation)
    """
    a_star = (r - r_e) / l
    return -E_c * (1 + a_star) * np.exp(-a_star)


def rose_force(r, E_c, r_e, l):
    """
    Force from Rose potential: F = -dE/dr
    """
    a_star = (r - r_e) / l
    # dE/dr = -E_c * (a*/l) * exp(-a*)
    return E_c * (a_star / l) * np.exp(-a_star)


def generate_material_parameters(n_materials=50, seed=42):
    """
    Generate random "materials" with different (E_c, r_e, l) parameters.

    These represent different chemical systems - the parameters vary
    but the underlying physics (Rose curve) is universal.
    """
    np.random.seed(seed)

    materials = []
    for i in range(n_materials):
        # Physically reasonable ranges
        E_c = np.random.uniform(0.5, 6.0)    # eV (cohesive energy)
        r_e = np.random.uniform(1.5, 4.0)    # Angstrom (bond length)
        l = np.random.uniform(0.3, 1.0)      # Angstrom (length scale)

        materials.append({
            'id': i,
            'E_c': E_c,
            'r_e': r_e,
            'l': l
        })

    return materials


def experiment1_universal_collapse(materials, save_path='fig1_universal_collapse.png'):
    """
    Experiment 1: Show that all binding curves collapse to ONE universal shape.

    This is the foundation - different materials have different E(r) curves,
    but when scaled properly, they all follow the same universal curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = cm.viridis(np.linspace(0, 1, len(materials)))

    # Panel 1: Raw E(r) curves - all different
    ax = axes[0]
    for mat, color in zip(materials, colors):
        r = np.linspace(0.5 * mat['r_e'], 3 * mat['r_e'], 200)
        E = rose_energy(r, mat['E_c'], mat['r_e'], mat['l'])
        ax.plot(r, E, color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('E(r) [eV]')
    ax.set_title('Raw binding curves\n(50 different "materials")')
    ax.set_ylim(-7, 1)

    # Panel 2: Scaled by E_c only (partial collapse)
    ax = axes[1]
    for mat, color in zip(materials, colors):
        r = np.linspace(0.5 * mat['r_e'], 3 * mat['r_e'], 200)
        E = rose_energy(r, mat['E_c'], mat['r_e'], mat['l'])
        E_scaled = E / mat['E_c']  # Scale energy by E_c
        ax.plot(r, E_scaled, color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel('r [Å]')
    ax.set_ylabel('E(r) / E_c')
    ax.set_title('Scaled by E_c only\n(still scattered - different r_e, l)')
    ax.set_ylim(-1.2, 0.3)

    # Panel 3: Fully scaled - UNIVERSAL COLLAPSE
    ax = axes[2]
    for mat, color in zip(materials, colors):
        r = np.linspace(0.5 * mat['r_e'], 3 * mat['r_e'], 200)
        a_star = (r - mat['r_e']) / mat['l']  # Reduced distance
        E = rose_energy(r, mat['E_c'], mat['r_e'], mat['l'])
        E_scaled = E / mat['E_c']
        ax.plot(a_star, E_scaled, color=color, alpha=0.5, linewidth=1)

    # Overlay the analytical universal curve
    a_star_theory = np.linspace(-1, 5, 200)
    E_theory = -(1 + a_star_theory) * np.exp(-a_star_theory)
    ax.plot(a_star_theory, E_theory, 'k--', linewidth=3, label='Universal: -(1+a*)exp(-a*)')

    ax.set_xlabel('a* = (r - r_e) / l')
    ax.set_ylabel('E / E_c')
    ax.set_title('UNIVERSAL COLLAPSE!\n(all curves follow ONE equation)')
    ax.set_ylim(-1.2, 0.3)
    ax.set_xlim(-1.5, 5)
    ax.legend(loc='lower right')
    ax.axvline(0, color='red', linestyle=':', alpha=0.5)
    ax.axhline(-1, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("EXPERIMENT 1: Universal Collapse")
    print("=" * 60)
    print("LEFT: 50 different E(r) curves - hard to learn directly")
    print("RIGHT: When scaled by (E_c, r_e, l), ALL collapse to ONE curve!")
    print()
    print("IMPLICATION: Instead of learning E(r) for each material,")
    print("learn the 3 adaptive parameters (E_c, r_e, l) and use")
    print("the universal Rose equation to compute E.")


def experiment2_parameter_learning(materials, save_path='fig2_ml_comparison.png'):
    """
    Experiment 2: Compare learning E(r) directly vs learning parameters.

    This is the key test - is it actually easier to learn (E_c, r_e, l)?
    """
    # Generate training data
    train_data = []

    for mat in materials:
        # Sample points along each curve
        r_values = np.linspace(0.6 * mat['r_e'], 2.5 * mat['r_e'], 30)

        for r in r_values:
            E = rose_energy(r, mat['E_c'], mat['r_e'], mat['l'])
            a_star = (r - mat['r_e']) / mat['l']

            train_data.append({
                'id': mat['id'],
                'r': r,
                'a_star': a_star,
                'E': E,
                'E_c': mat['E_c'],
                'r_e': mat['r_e'],
                'l': mat['l']
            })

    import pandas as pd
    df = pd.DataFrame(train_data)

    # Split by material ID (not random points!)
    unique_ids = df['id'].unique()
    n_train = int(0.8 * len(unique_ids))
    train_ids = unique_ids[:n_train]
    test_ids = unique_ids[n_train:]

    train_df = df[df['id'].isin(train_ids)]
    test_df = df[df['id'].isin(test_ids)]

    print(f"Training on {len(train_ids)} materials, testing on {len(test_ids)} materials")

    # =========================================
    # APPROACH A: Learn E(r) directly
    # Input: (r, some descriptor of the material)
    # Output: E
    # =========================================

    # Use (r, E_c, r_e, l) as input - this is "cheating" but shows best case
    X_train_A = train_df[['r', 'E_c', 'r_e', 'l']].values
    y_train_A = train_df['E'].values
    X_test_A = test_df[['r', 'E_c', 'r_e', 'l']].values
    y_test_A = test_df['E'].values

    scaler_A = StandardScaler()
    X_train_A_scaled = scaler_A.fit_transform(X_train_A)
    X_test_A_scaled = scaler_A.transform(X_test_A)

    model_A = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=2000,
                           random_state=42, early_stopping=True)
    model_A.fit(X_train_A_scaled, y_train_A)
    y_pred_A = model_A.predict(X_test_A_scaled)

    rmse_A = np.sqrt(mean_squared_error(y_test_A, y_pred_A))
    r2_A = r2_score(y_test_A, y_pred_A)

    # =========================================
    # APPROACH B: Learn in reduced coordinates
    # Input: a* = (r - r_e) / l
    # Output: E / E_c (scaled energy)
    # Then reconstruct E = E_c * (E/E_c)
    # =========================================

    X_train_B = train_df[['a_star']].values
    y_train_B = (train_df['E'] / train_df['E_c']).values
    X_test_B = test_df[['a_star']].values
    y_test_B_scaled = (test_df['E'] / test_df['E_c']).values

    scaler_B = StandardScaler()
    X_train_B_scaled = scaler_B.fit_transform(X_train_B)
    X_test_B_scaled = scaler_B.transform(X_test_B)

    model_B = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000,
                           random_state=42, early_stopping=True)
    model_B.fit(X_train_B_scaled, y_train_B)
    y_pred_B_scaled = model_B.predict(X_test_B_scaled)

    # Reconstruct actual energy
    y_pred_B = y_pred_B_scaled * test_df['E_c'].values

    rmse_B = np.sqrt(mean_squared_error(y_test_A, y_pred_B))
    r2_B = r2_score(y_test_A, y_pred_B)

    # =========================================
    # APPROACH C: Use analytical Rose equation
    # Just plug in the parameters - no ML needed!
    # =========================================

    y_pred_C = rose_energy(test_df['r'].values, test_df['E_c'].values,
                           test_df['r_e'].values, test_df['l'].values)

    rmse_C = np.sqrt(mean_squared_error(y_test_A, y_pred_C))
    r2_C = r2_score(y_test_A, y_pred_C)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(y_test_A, y_pred_A, alpha=0.5, s=20, c='blue')
    lim = [min(y_test_A.min(), y_pred_A.min()), max(y_test_A.max(), y_pred_A.max())]
    ax.plot(lim, lim, 'r--', linewidth=2)
    ax.set_xlabel('True E [eV]')
    ax.set_ylabel('Predicted E [eV]')
    ax.set_title(f'Approach A: Learn E(r) directly\nRMSE = {rmse_A:.4f} eV, R² = {r2_A:.4f}')
    ax.set_aspect('equal')

    ax = axes[1]
    ax.scatter(y_test_A, y_pred_B, alpha=0.5, s=20, c='green')
    ax.plot(lim, lim, 'r--', linewidth=2)
    ax.set_xlabel('True E [eV]')
    ax.set_ylabel('Predicted E [eV]')
    ax.set_title(f'Approach B: Learn E/E_c(a*)\nRMSE = {rmse_B:.4f} eV, R² = {r2_B:.4f}')
    ax.set_aspect('equal')

    ax = axes[2]
    ax.scatter(y_test_A, y_pred_C, alpha=0.5, s=20, c='orange')
    ax.plot(lim, lim, 'r--', linewidth=2)
    ax.set_xlabel('True E [eV]')
    ax.set_ylabel('Predicted E [eV]')
    ax.set_title(f'Approach C: Rose equation (no ML!)\nRMSE = {rmse_C:.4f} eV, R² = {r2_C:.4f}')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 2: ML Comparison")
    print("=" * 60)
    print(f"Approach A (Learn E directly):    RMSE = {rmse_A:.4f} eV, R² = {r2_A:.4f}")
    print(f"Approach B (Learn E/E_c vs a*):   RMSE = {rmse_B:.4f} eV, R² = {r2_B:.4f}")
    print(f"Approach C (Rose equation):       RMSE = {rmse_C:.4f} eV, R² = {r2_C:.4f}")
    print()
    print("KEY INSIGHT: If we know the adaptive parameters (E_c, r_e, l),")
    print("we don't need ML at all - just use the universal equation!")

    return df


def experiment3_parameter_prediction(materials, save_path='fig3_parameter_prediction.png'):
    """
    Experiment 3: The REAL task - predict (E_c, r_e, l) from local environment.

    In a real application:
    - Input: Local atomic environment (descriptors)
    - Output: (E_c, r_e, l) for that environment
    - Energy: Compute from Rose equation

    This is analogous to aPBE0 where we predict α from density.
    """
    # Simulate "local environment descriptors"
    # In reality these would be SOAP, ACE, symmetry functions, etc.
    # Here we just make up some correlated features

    np.random.seed(42)

    data = []
    for mat in materials:
        # Fake "descriptors" that correlate with parameters
        # (In reality, these come from atomic positions)
        coordination = 4 + 2 * np.random.randn()  # Coordination number
        electronegativity = 1.5 + 0.5 * np.random.randn()
        atomic_radius = mat['r_e'] / 2.5 + 0.1 * np.random.randn()

        data.append({
            'id': mat['id'],
            'coordination': coordination,
            'electronegativity': electronegativity,
            'atomic_radius': atomic_radius,
            'E_c': mat['E_c'],
            'r_e': mat['r_e'],
            'l': mat['l']
        })

    import pandas as pd
    df = pd.DataFrame(data)

    # Split
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # Train models to predict each parameter
    X_train = train_df[['coordination', 'electronegativity', 'atomic_radius']].values
    X_test = test_df[['coordination', 'electronegativity', 'atomic_radius']].values

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    results = {}
    for param in ['E_c', 'r_e', 'l']:
        y_train = train_df[param].values
        y_test = test_df[param].values

        model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[param] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'rmse': rmse,
            'r2': r2
        }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (param, res) in zip(axes, results.items()):
        ax.scatter(res['y_test'], res['y_pred'], alpha=0.7, s=50)
        lim = [min(res['y_test'].min(), res['y_pred'].min()),
               max(res['y_test'].max(), res['y_pred'].max())]
        ax.plot(lim, lim, 'r--', linewidth=2)
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'Predicting {param}\nRMSE = {res["rmse"]:.3f}, R² = {res["r2"]:.3f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 3: Parameter Prediction")
    print("=" * 60)
    print("Predicting adaptive parameters from local environment:")
    for param, res in results.items():
        print(f"  {param}: RMSE = {res['rmse']:.3f}, R² = {res['r2']:.3f}")
    print()
    print("ANALOGY TO aPBE0:")
    print("  aPBE0: descriptor → α → E = (1-α)E_PBE + α*E_HF")
    print("  Here:  descriptor → (E_c, r_e, l) → E = Rose(r; E_c, r_e, l)")

    return results


def experiment4_apbe0_comparison(save_path='fig4_apbe0_analogy.png'):
    """
    Experiment 4: Visual comparison of aPBE0 logic vs Rose/UBER logic.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: aPBE0 schematic
    ax = axes[0]
    ax.text(0.5, 0.95, 'aPBE0 Approach', fontsize=16, fontweight='bold',
            ha='center', transform=ax.transAxes)

    text_apbe0 = """
Standard DFT:
    E = E_PBE  or  E = E_HF
    (neither is universally accurate)

Hybrid DFT:
    E = (1-α)E_PBE + α·E_HF
    (α = 0.25 is a global constant)

aPBE0 (Your previous work):
    α = ML(local density features)
    E = (1-α)E_PBE + α·E_HF

Key insight:
    • α is bounded [0, 1]
    • α varies smoothly with environment
    • α is more transferable than E
    • Physics equation does the heavy lifting
"""
    ax.text(0.05, 0.85, text_apbe0, fontsize=11, family='monospace',
            va='top', transform=ax.transAxes)
    ax.axis('off')

    # Right: Rose/UBER schematic
    ax = axes[1]
    ax.text(0.5, 0.95, 'Rose/UBER Approach', fontsize=16, fontweight='bold',
            ha='center', transform=ax.transAxes)

    text_rose = """
Standard ML Potential:
    E = NN(r, descriptors)
    (learns everything from scratch)

Rose Universal Equation:
    E = -E_c·(1 + a*)·exp(-a*)
    where a* = (r - r_e) / l
    (universal shape, material-specific parameters)

Adaptive Rose (This work):
    (E_c, r_e, l) = ML(local environment)
    E = Rose(r; E_c, r_e, l)

Key insight:
    • Parameters are bounded & physical
    • Parameters vary smoothly
    • Parameters are transferable
    • Physics equation does the heavy lifting
"""
    ax.text(0.05, 0.85, text_rose, fontsize=11, family='monospace',
            va='top', transform=ax.transAxes)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\nEXPERIMENT 4: The aPBE0 Analogy")
    print("=" * 60)
    print("Both approaches share the same philosophy:")
    print("  1. Don't learn the target quantity directly")
    print("  2. Learn adaptive PARAMETERS that are simpler")
    print("  3. Use PHYSICS EQUATIONS to compute the target")
    print("  4. Parameters are bounded, smooth, transferable")


def main():
    print("=" * 70)
    print("ADAPTIVE PARAMETER LEARNING FOR INTERATOMIC POTENTIALS")
    print("=" * 70)
    print()
    print("Demonstrating the Rose/UBER approach - analogous to aPBE0")
    print()
    print("Core idea: Learn (E_c, r_e, l) instead of E(r) directly,")
    print("then use the universal Rose equation to compute energy.")
    print()

    # Generate materials
    materials = generate_material_parameters(n_materials=50)

    # Run experiments
    experiment1_universal_collapse(materials)
    experiment2_parameter_learning(materials)
    experiment3_parameter_prediction(materials)
    experiment4_apbe0_comparison()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The "Adaptive Parameter" Approach:

1. OBSERVATION: All binding curves follow the universal Rose equation
   E(a*) = -E_c * (1 + a*) * exp(-a*)

2. IMPLICATION: Instead of learning E(r) directly (hard, varies wildly),
   learn the adaptive parameters (E_c, r_e, l) which are:
   - Bounded (physically constrained)
   - Smooth (vary predictably with environment)
   - Transferable (same parameters work across similar systems)

3. CONNECTION TO aPBE0:
   aPBE0: descriptor → α → E via physics equation
   Rose:  descriptor → (E_c, r_e, l) → E via physics equation

4. ADVANTAGE: The physics equation enforces correct behavior
   (dissociation, equilibrium, curvature) automatically.
   ML only needs to learn 3 simple parameters per environment.

NEXT STEPS:
- Apply to real DFT binding curves (bulk metals, molecules)
- Compare transferability: new materials, new conditions
- Connect to Anatole's work on density-based descriptors
""")


if __name__ == "__main__":
    main()
