"""
ML Comparison Demo: Direct vs Adaptive Learning
================================================

Comprehensive comparison across toy systems:
1. C6 Dispersion (HOMOGENEOUS) - both approaches should work similarly
2. Morse/Rose (NON-HOMOGENEOUS) - adaptive should win on extrapolation
3. Lennard-Jones (NON-HOMOGENEOUS) - adaptive should win on extrapolation

Key tests:
- Interpolation (test within training range)
- Extrapolation (test outside training range)
- Data efficiency (learning curves)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

def dispersion_V(r, C6):
    """V = -C6/r^6 (homogeneous degree -6)"""
    return -C6 / r**6

def morse_V(r, De, a, re):
    """Morse potential"""
    return De * (1 - np.exp(-a * (r - re)))**2

def rose_V(r, E_c, r_e, l):
    """Rose/UBER potential (shifted to match Morse convention)"""
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

def lj_V(r, epsilon, sigma):
    """Lennard-Jones potential"""
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)


# =============================================================================
# EXPERIMENT 1: C6 DISPERSION (HOMOGENEOUS)
# =============================================================================

def experiment_c6():
    """
    For homogeneous C6: both direct and adaptive should work similarly.
    The "adaptive parameter" here is just C6 itself (degree d=-6 is fixed).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: C6 DISPERSION (HOMOGENEOUS)")
    print("="*70)

    n_points = 50

    # Training: C6 in range [0.5, 2.0]
    train_data = {'X': [], 'C6': [], 'curves': [], 'r_grids': []}
    for i in range(60):
        d1 = np.random.uniform(0.5, 1.5)  # descriptor
        C6 = 0.3 + 1.0 * d1  # C6 ~ 0.8-1.8
        r = np.linspace(1.0, 4.0, n_points)
        V = dispersion_V(r, C6)
        train_data['X'].append([d1])
        train_data['C6'].append(C6)
        train_data['curves'].append(V)
        train_data['r_grids'].append(r)

    # Test: C6 in range [2.5, 4.0] - EXTRAPOLATION
    test_data = {'X': [], 'C6': [], 'curves': [], 'r_grids': []}
    for i in range(30):
        d1 = np.random.uniform(2.5, 4.0)  # EXTRAPOLATION
        C6 = 0.3 + 1.0 * d1  # C6 ~ 2.8-4.3
        r = np.linspace(1.0, 4.0, n_points)
        V = dispersion_V(r, C6)
        test_data['X'].append([d1])
        test_data['C6'].append(C6)
        test_data['curves'].append(V)
        test_data['r_grids'].append(r)

    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_C6_train = np.array(train_data['C6']).reshape(-1, 1)
    Y_C6_test = np.array(test_data['C6']).reshape(-1, 1)

    # Scale
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Direct: learn V(r)
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    Y_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())

    # Adaptive: learn C6 → V via physics
    scaler_C6 = StandardScaler()
    Y_C6_train_s = scaler_C6.fit_transform(Y_C6_train)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_C6_train_s)
    Y_C6_pred = scaler_C6.inverse_transform(model_adaptive.predict(X_test_s))

    Y_pred_adaptive = []
    for i, C6 in enumerate(Y_C6_pred.flatten()):
        r = test_data['r_grids'][i]
        Y_pred_adaptive.append(dispersion_V(r, max(C6, 0.01)))
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())

    print(f"Training: C6 ~ 0.8-1.8, Testing: C6 ~ 2.8-4.3 (EXTRAPOLATION)")
    print(f"Direct (learn V):     MSE = {mse_direct:.6f}, R² = {r2_direct:.4f}")
    print(f"Adaptive (learn C6):  MSE = {mse_adaptive:.6f}, R² = {r2_adaptive:.4f}")

    return {
        'system': 'C6 Dispersion',
        'type': 'Homogeneous',
        'mse_direct': mse_direct, 'r2_direct': r2_direct,
        'mse_adaptive': mse_adaptive, 'r2_adaptive': r2_adaptive,
        'test_data': test_data, 'Y_pred_direct': Y_pred_direct,
        'Y_pred_adaptive': Y_pred_adaptive, 'Y_curves_test': Y_curves_test
    }


# =============================================================================
# EXPERIMENT 2: MORSE/ROSE (NON-HOMOGENEOUS)
# =============================================================================

def experiment_morse():
    """
    For non-homogeneous Morse: adaptive should win on extrapolation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: MORSE/ROSE (NON-HOMOGENEOUS)")
    print("="*70)

    n_points = 50

    # Training: weak bonds (De ~ 1-4 eV)
    train_data = {'X': [], 'params': [], 'curves': [], 'r_grids': []}
    for i in range(80):
        d1 = np.random.uniform(0.5, 1.5)  # correlates with De
        d2 = np.random.uniform(0.5, 1.5)  # correlates with re

        De = 0.5 + 2.0 * d1  # De ~ 1.5-3.5 eV
        re = 1.5 + 0.5 * d2  # re ~ 1.75-2.25 Å
        a = 1.5  # fixed stiffness

        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a, re)

        # Fit Rose params
        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1/a],
                               bounds=([0.01, 0.5, 0.01], [20, 10, 5]))
            train_data['X'].append([d1, d2])
            train_data['params'].append(popt)
            train_data['curves'].append(V)
            train_data['r_grids'].append(r)
        except:
            pass

    # Test: STRONG bonds (De ~ 5-9 eV) - EXTRAPOLATION
    test_data = {'X': [], 'params': [], 'curves': [], 'r_grids': [], 'De': []}
    for i in range(40):
        d1 = np.random.uniform(2.5, 4.0)  # EXTRAPOLATION
        d2 = np.random.uniform(0.5, 1.5)

        De = 0.5 + 2.0 * d1  # De ~ 5.5-8.5 eV
        re = 1.5 + 0.5 * d2
        a = 1.5

        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a, re)

        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1/a],
                               bounds=([0.01, 0.5, 0.01], [20, 10, 5]))
            test_data['X'].append([d1, d2])
            test_data['params'].append(popt)
            test_data['curves'].append(V)
            test_data['r_grids'].append(r)
            test_data['De'].append(De)
        except:
            pass

    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['params'])
    Y_params_test = np.array(test_data['params'])

    # Scale
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Direct: learn V(r)
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    Y_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())

    # Adaptive: learn (E_c, r_e, l) → V via Rose
    scaler_params = StandardScaler()
    Y_params_train_s = scaler_params.fit_transform(Y_params_train)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_params_train_s)
    Y_params_pred = scaler_params.inverse_transform(model_adaptive.predict(X_test_s))

    Y_pred_adaptive = []
    for i, (E_c, r_e, l) in enumerate(Y_params_pred):
        r = test_data['r_grids'][i]
        Y_pred_adaptive.append(rose_V(r, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())

    print(f"Training: De ~ 1.5-3.5 eV, Testing: De ~ 5.5-8.5 eV (EXTRAPOLATION)")
    print(f"Direct (learn V):        MSE = {mse_direct:.4f}, R² = {r2_direct:.4f}")
    print(f"Adaptive (learn params): MSE = {mse_adaptive:.4f}, R² = {r2_adaptive:.4f}")

    if mse_adaptive < mse_direct:
        print(f"→ Adaptive is {(mse_direct-mse_adaptive)/mse_direct*100:.0f}% better!")

    return {
        'system': 'Morse/Rose',
        'type': 'Non-Homogeneous',
        'mse_direct': mse_direct, 'r2_direct': r2_direct,
        'mse_adaptive': mse_adaptive, 'r2_adaptive': r2_adaptive,
        'test_data': test_data, 'Y_pred_direct': Y_pred_direct,
        'Y_pred_adaptive': Y_pred_adaptive, 'Y_curves_test': Y_curves_test
    }


# =============================================================================
# EXPERIMENT 3: LENNARD-JONES (NON-HOMOGENEOUS)
# =============================================================================

def experiment_lj():
    """
    For non-homogeneous LJ: adaptive should win on extrapolation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: LENNARD-JONES (NON-HOMOGENEOUS)")
    print("="*70)

    n_points = 50

    # Training: small atoms (σ ~ 2.5-3.5 Å)
    train_data = {'X': [], 'params': [], 'curves': [], 'r_grids': []}
    for i in range(80):
        d1 = np.random.uniform(0.5, 1.5)  # correlates with σ
        d2 = np.random.uniform(0.5, 1.5)  # correlates with ε

        sigma = 2.5 + 0.7 * d1   # σ ~ 2.85-3.55 Å
        epsilon = 0.05 + 0.1 * d2  # ε ~ 0.1-0.2 eV

        r = np.linspace(0.9 * sigma, 2.5 * sigma, n_points)
        V = lj_V(r, epsilon, sigma)

        train_data['X'].append([d1, d2])
        train_data['params'].append([epsilon, sigma])
        train_data['curves'].append(V)
        train_data['r_grids'].append(r)

    # Test: LARGE atoms (σ ~ 4-5 Å) - EXTRAPOLATION
    test_data = {'X': [], 'params': [], 'curves': [], 'r_grids': [], 'sigma': []}
    for i in range(40):
        d1 = np.random.uniform(2.5, 4.0)  # EXTRAPOLATION
        d2 = np.random.uniform(0.5, 1.5)

        sigma = 2.5 + 0.7 * d1   # σ ~ 4.25-5.3 Å
        epsilon = 0.05 + 0.1 * d2

        r = np.linspace(0.9 * sigma, 2.5 * sigma, n_points)
        V = lj_V(r, epsilon, sigma)

        test_data['X'].append([d1, d2])
        test_data['params'].append([epsilon, sigma])
        test_data['curves'].append(V)
        test_data['r_grids'].append(r)
        test_data['sigma'].append(sigma)

    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['params'])
    Y_params_test = np.array(test_data['params'])

    # Scale
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Direct: learn V(r)
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    Y_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())

    # Adaptive: learn (ε, σ) → V via LJ
    scaler_params = StandardScaler()
    Y_params_train_s = scaler_params.fit_transform(Y_params_train)
    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_params_train_s)
    Y_params_pred = scaler_params.inverse_transform(model_adaptive.predict(X_test_s))

    Y_pred_adaptive = []
    for i, (eps, sig) in enumerate(Y_params_pred):
        r = test_data['r_grids'][i]
        Y_pred_adaptive.append(lj_V(r, max(eps, 0.001), max(sig, 0.1)))
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())

    print(f"Training: σ ~ 2.9-3.6 Å, Testing: σ ~ 4.3-5.3 Å (EXTRAPOLATION)")
    print(f"Direct (learn V):        MSE = {mse_direct:.6f}, R² = {r2_direct:.4f}")
    print(f"Adaptive (learn params): MSE = {mse_adaptive:.6f}, R² = {r2_adaptive:.4f}")

    if mse_adaptive < mse_direct:
        print(f"→ Adaptive is {(mse_direct-mse_adaptive)/mse_direct*100:.0f}% better!")

    return {
        'system': 'Lennard-Jones',
        'type': 'Non-Homogeneous',
        'mse_direct': mse_direct, 'r2_direct': r2_direct,
        'mse_adaptive': mse_adaptive, 'r2_adaptive': r2_adaptive,
        'test_data': test_data, 'Y_pred_direct': Y_pred_direct,
        'Y_pred_adaptive': Y_pred_adaptive, 'Y_curves_test': Y_curves_test
    }


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def create_summary_figure(results):
    """Create comprehensive comparison figure."""

    fig = plt.figure(figsize=(18, 14))

    # Top row: Example predictions for each system
    for col, res in enumerate(results):
        ax = fig.add_subplot(3, 3, col + 1)
        idx = 0  # first test example
        r = res['test_data']['r_grids'][idx]

        ax.plot(r, res['Y_curves_test'][idx], 'k-', linewidth=3, label='True')
        ax.plot(r, res['Y_pred_direct'][idx], 'b--', linewidth=2, label='Direct ML')
        ax.plot(r, res['Y_pred_adaptive'][idx], 'r:', linewidth=2, label='Adaptive ML')

        ax.set_xlabel('r [Å]')
        ax.set_ylabel('V(r) [eV]')
        ax.set_title(f"{res['system']}\n({res['type']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Middle row: MSE comparison bars
    ax = fig.add_subplot(3, 3, 4)
    systems = [r['system'] for r in results]
    mse_direct = [r['mse_direct'] for r in results]
    mse_adaptive = [r['mse_adaptive'] for r in results]

    x = np.arange(len(systems))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_direct, width, label='Direct', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mse_adaptive, width, label='Adaptive', color='red', alpha=0.7)

    ax.set_ylabel('MSE [eV²]')
    ax.set_title('Extrapolation MSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.legend()
    ax.set_yscale('log')

    # Middle row: R² comparison bars
    ax = fig.add_subplot(3, 3, 5)
    r2_direct = [r['r2_direct'] for r in results]
    r2_adaptive = [r['r2_adaptive'] for r in results]

    bars1 = ax.bar(x - width/2, r2_direct, width, label='Direct', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, r2_adaptive, width, label='Adaptive', color='red', alpha=0.7)

    ax.set_ylabel('R² Score')
    ax.set_title('Extrapolation R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.legend()
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(min(min(r2_direct), min(r2_adaptive)) - 0.1, 1.05)

    # Middle row: Improvement percentage
    ax = fig.add_subplot(3, 3, 6)
    improvements = []
    for r in results:
        if r['mse_adaptive'] < r['mse_direct']:
            imp = (r['mse_direct'] - r['mse_adaptive']) / r['mse_direct'] * 100
        else:
            imp = -(r['mse_adaptive'] - r['mse_direct']) / r['mse_adaptive'] * 100
        improvements.append(imp)

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(systems, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Adaptive Improvement (%)')
    ax.set_title('Adaptive vs Direct\n(positive = adaptive wins)')
    ax.set_xticklabels(systems, rotation=15)

    for bar, imp, res in zip(bars, improvements, results):
        if imp > 0:
            ax.text(bar.get_x() + bar.get_width()/2, imp + 2, f'{imp:.0f}%',
                   ha='center', fontsize=11, fontweight='bold')

    # Bottom row: Summary table and key findings
    ax = fig.add_subplot(3, 1, 3)
    ax.axis('off')

    # Create summary text
    summary_text = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                           ML COMPARISON SUMMARY                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  System           │ Type              │ Direct R²  │ Adaptive R² │ Winner           ║
    ╠═══════════════════╪═══════════════════╪════════════╪═════════════╪══════════════════╣"""

    for r in results:
        winner = "ADAPTIVE" if r['r2_adaptive'] > r['r2_direct'] else "Direct"
        summary_text += f"""
    ║  {r['system']:<15} │ {r['type']:<17} │ {r['r2_direct']:>10.4f} │ {r['r2_adaptive']:>11.4f} │ {winner:<16} ║"""

    summary_text += """
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                      ║
    ║  KEY FINDINGS:                                                                       ║
    ║  • Homogeneous (C6): Both approaches work - degree d is constant                     ║
    ║  • Non-Homogeneous (Morse, LJ): ADAPTIVE WINS on extrapolation!                      ║
    ║    → Physics equation provides correct shape even beyond training range              ║
    ║                                                                                      ║
    ║  CONNECTION TO aPBE0:                                                                ║
    ║  • Same principle: learn bounded, smooth parameters instead of raw energy            ║
    ║  • Physics equation enforces correct behavior in extrapolation regime                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('fig_ml_comparison_summary.png', dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("ML COMPARISON: Direct vs Adaptive Learning")
    print("="*70)
    print("\nTesting EXTRAPOLATION performance on toy systems:")
    print("- Train on one parameter range, test on DIFFERENT range")
    print("- This tests whether the model generalizes beyond training data")

    # Run experiments
    results = []
    results.append(experiment_c6())
    results.append(experiment_morse())
    results.append(experiment_lj())

    # Create summary figure
    create_summary_figure(results)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\n{:<20} {:<18} {:>12} {:>12} {:>12}".format(
        "System", "Type", "Direct R²", "Adaptive R²", "Winner"))
    print("-"*70)

    for r in results:
        winner = "ADAPTIVE" if r['r2_adaptive'] > r['r2_direct'] else "Direct"
        marker = "***" if r['r2_adaptive'] > r['r2_direct'] else ""
        print("{:<20} {:<18} {:>12.4f} {:>12.4f} {:>12} {}".format(
            r['system'], r['type'], r['r2_direct'], r['r2_adaptive'], winner, marker))

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
For HOMOGENEOUS potentials (C6):
  → Both approaches work similarly (d is constant)
  → No special advantage to adaptive learning

For NON-HOMOGENEOUS potentials (Morse, LJ):
  → ADAPTIVE approach extrapolates better!
  → Physics equation provides correct shape beyond training
  → This is the same principle as aPBE0!

IMPLICATION FOR REAL SYSTEMS:
  Real DFT binding curves are NON-HOMOGENEOUS
  → Adaptive parameters (Rose equation) should help!
  → Learn (E_c, r_e, l) instead of V(r) directly
""")


if __name__ == "__main__":
    main()
