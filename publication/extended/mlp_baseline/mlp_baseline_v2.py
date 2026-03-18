"""
MLP nonlinear baseline comparison v2: More thorough sweep.

Addresses reviewer concern more rigorously by:
  1. Testing multiple activations (relu, tanh, logistic)
  2. Testing more architectures (2-4 layers)
  3. Testing MLP in BOTH direct and adaptive pathways
  4. Giving the direct approach every possible advantage

The key comparison is:
  - Direct Ridge vs Direct MLP (best) vs Adaptive Ridge vs Adaptive MLP (best)

If the advantage comes from physics (not model class), then:
  Adaptive Ridge > Direct MLP (best)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import curve_fit
import warnings
import itertools
warnings.filterwarnings('ignore')

# =============================================================================
# Potentials (same as before)
# =============================================================================

def lennard_jones(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def morse_V(r, De, a, re):
    return De * (1 - np.exp(-a * (r - re)))**2

def rose_V(r, E_c, r_e, l):
    a_star = (r - r_e) / l
    return E_c * (1 - (1 + a_star) * np.exp(-a_star))

# =============================================================================
# Dataset generation
# =============================================================================

def generate_lj_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r = np.linspace(2.5, 12.0, n_points)
    X, params, curves = [], [], []
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2
        X.append([d1, d2])
        params.append([epsilon, sigma])
        curves.append(lennard_jones(r, epsilon, sigma))
    return {'X': np.array(X), 'params': np.array(params),
            'curves': np.array(curves), 'r': r}

def generate_rose_dataset(n_samples, d1_range, d2_range, n_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, morse_params, rose_params, curves, r_grids = [], [], [], [], []
    a_fixed = 1.5
    for _ in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        De = 0.5 + 2.0 * d1
        re = 1.5 + 0.5 * d2
        r = np.linspace(0.7 * re, 3.0 * re, n_points)
        V = morse_V(r, De, a_fixed, re)
        try:
            popt, _ = curve_fit(rose_V, r, V, p0=[De, re, 1.0 / a_fixed],
                                bounds=([0.01, 0.5, 0.01], [20, 10, 5]), maxfev=5000)
            X.append([d1, d2])
            morse_params.append([De, re, a_fixed])
            rose_params.append(popt)
            curves.append(V)
            r_grids.append(r)
        except RuntimeError:
            pass
    return {'X': np.array(X), 'morse_params': np.array(morse_params),
            'rose_params': np.array(rose_params), 'curves': np.array(curves),
            'r_grids': r_grids}

# =============================================================================
# MLP configurations: full sweep
# =============================================================================

ACTIVATIONS = ['relu', 'tanh', 'logistic']
ARCHITECTURES = [
    (32, 32),
    (64, 64),
    (128, 64),
    (256, 128),
    (64, 64, 64),
    (128, 64, 32),
    (128, 128, 64, 32),  # 4-layer
]

def make_mlp(hidden_layers, activation, max_iter=3000):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        alpha=0.001,
        random_state=42,
    )

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_all_lj(train_data, test_data):
    """Evaluate all model variants on LJ data."""
    X_train, X_test = train_data['X'], test_data['X']
    Y_train, Y_test = train_data['curves'], test_data['curves']
    P_train, P_test = train_data['params'], test_data['params']
    r = train_data['r']

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct Ridge ---
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    m = Ridge(alpha=1.0).fit(X_tr, Y_tr_s)
    Y_pred = scaler_V.inverse_transform(m.predict(X_te))
    results['Direct Ridge'] = mean_absolute_error(Y_test, Y_pred)

    # --- 2. Direct MLP (sweep) ---
    best_direct_mlp = np.inf
    best_direct_mlp_name = ''
    for arch in ARCHITECTURES:
        for act in ACTIVATIONS:
            name = f'Direct MLP {arch} {act}'
            scaler_V2 = StandardScaler()
            Y_tr_s2 = scaler_V2.fit_transform(Y_train)
            mlp = make_mlp(arch, act)
            mlp.fit(X_tr, Y_tr_s2)
            Y_pred = scaler_V2.inverse_transform(mlp.predict(X_te))
            mae = mean_absolute_error(Y_test, Y_pred)
            results[name] = mae
            if mae < best_direct_mlp:
                best_direct_mlp = mae
                best_direct_mlp_name = name

    # --- 3. Adaptive Ridge ---
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(P_train)
    m = Ridge(alpha=1.0).fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(m.predict(X_te))
    Y_pred = np.array([lennard_jones(r, max(e, 0.001), max(s, 0.1)) for e, s in P_pred])
    results['Adaptive Ridge'] = mean_absolute_error(Y_test, Y_pred)

    # --- 4. Adaptive MLP (sweep) ---
    best_adapt_mlp = np.inf
    best_adapt_mlp_name = ''
    for arch in ARCHITECTURES:
        for act in ACTIVATIONS:
            name = f'Adaptive MLP {arch} {act}'
            scaler_P2 = StandardScaler()
            P_tr_s2 = scaler_P2.fit_transform(P_train)
            mlp = make_mlp(arch, act)
            mlp.fit(X_tr, P_tr_s2)
            P_pred = scaler_P2.inverse_transform(mlp.predict(X_te))
            Y_pred = np.array([lennard_jones(r, max(e, 0.001), max(s, 0.1))
                               for e, s in P_pred])
            mae = mean_absolute_error(Y_test, Y_pred)
            results[name] = mae
            if mae < best_adapt_mlp:
                best_adapt_mlp = mae
                best_adapt_mlp_name = name

    results['_best_direct_mlp'] = best_direct_mlp_name
    results['_best_direct_mlp_mae'] = best_direct_mlp
    results['_best_adapt_mlp'] = best_adapt_mlp_name
    results['_best_adapt_mlp_mae'] = best_adapt_mlp
    return results


def evaluate_all_rose(train_data, test_data):
    """Evaluate all model variants on Rose/UBER data."""
    X_train, X_test = train_data['X'], test_data['X']
    Y_train, Y_test = train_data['curves'], test_data['curves']
    P_train = train_data['rose_params']

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)

    results = {}

    # --- 1. Direct Ridge ---
    scaler_V = StandardScaler()
    Y_tr_s = scaler_V.fit_transform(Y_train)
    m = Ridge(alpha=1.0).fit(X_tr, Y_tr_s)
    Y_pred = scaler_V.inverse_transform(m.predict(X_te))
    results['Direct Ridge'] = mean_absolute_error(Y_test, Y_pred)

    # --- 2. Direct MLP (sweep) ---
    best_direct_mlp = np.inf
    best_direct_mlp_name = ''
    for arch in ARCHITECTURES:
        for act in ACTIVATIONS:
            name = f'Direct MLP {arch} {act}'
            scaler_V2 = StandardScaler()
            Y_tr_s2 = scaler_V2.fit_transform(Y_train)
            mlp = make_mlp(arch, act)
            mlp.fit(X_tr, Y_tr_s2)
            Y_pred = scaler_V2.inverse_transform(mlp.predict(X_te))
            mae = mean_absolute_error(Y_test, Y_pred)
            results[name] = mae
            if mae < best_direct_mlp:
                best_direct_mlp = mae
                best_direct_mlp_name = name

    # --- 3. Adaptive Ridge ---
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(P_train)
    m = Ridge(alpha=1.0).fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(m.predict(X_te))
    Y_pred = []
    for i, (E_c, r_e, l) in enumerate(P_pred):
        r_i = test_data['r_grids'][i]
        Y_pred.append(rose_V(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
    results['Adaptive Ridge'] = mean_absolute_error(Y_test, np.array(Y_pred))

    # --- 4. Adaptive MLP (sweep) ---
    best_adapt_mlp = np.inf
    best_adapt_mlp_name = ''
    for arch in ARCHITECTURES:
        for act in ACTIVATIONS:
            name = f'Adaptive MLP {arch} {act}'
            scaler_P2 = StandardScaler()
            P_tr_s2 = scaler_P2.fit_transform(P_train)
            mlp = make_mlp(arch, act)
            mlp.fit(X_tr, P_tr_s2)
            P_pred = scaler_P2.inverse_transform(mlp.predict(X_te))
            Y_pred = []
            for i, (E_c, r_e, l) in enumerate(P_pred):
                r_i = test_data['r_grids'][i]
                Y_pred.append(rose_V(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
            mae = mean_absolute_error(Y_test, np.array(Y_pred))
            results[name] = mae
            if mae < best_adapt_mlp:
                best_adapt_mlp = mae
                best_adapt_mlp_name = name

    results['_best_direct_mlp'] = best_direct_mlp_name
    results['_best_direct_mlp_mae'] = best_direct_mlp
    results['_best_adapt_mlp'] = best_adapt_mlp_name
    results['_best_adapt_mlp_mae'] = best_adapt_mlp
    return results


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(system, regime, n_train=160, n_seeds=5):
    """Run full activation/architecture sweep."""
    # Collect per-seed results for the 4 summary methods
    summary_methods = ['Direct Ridge', 'Best Direct MLP', 'Adaptive Ridge', 'Best Adaptive MLP']
    seed_results = {m: [] for m in summary_methods}
    # Also track per-config results across seeds
    all_configs = {}

    for s in range(n_seeds):
        seed = s * 100 + n_train
        if system == 'lj':
            train = generate_lj_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
            if regime == 'extrap':
                test = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=seed + 10000)
            else:
                test = generate_lj_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
            res = evaluate_all_lj(train, test)
        else:
            train = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
            if regime == 'extrap':
                test = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
            else:
                test = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
            res = evaluate_all_rose(train, test)

        seed_results['Direct Ridge'].append(res['Direct Ridge'])
        seed_results['Best Direct MLP'].append(res['_best_direct_mlp_mae'])
        seed_results['Adaptive Ridge'].append(res['Adaptive Ridge'])
        seed_results['Best Adaptive MLP'].append(res['_best_adapt_mlp_mae'])

        # Store all config results
        for k, v in res.items():
            if not k.startswith('_') and isinstance(v, float):
                if k not in all_configs:
                    all_configs[k] = []
                all_configs[k].append(v)

        print(f"  Seed {s}: Direct Ridge={res['Direct Ridge']:.4f}, "
              f"Best Direct MLP={res['_best_direct_mlp_mae']:.4f} ({res['_best_direct_mlp']}), "
              f"Adaptive Ridge={res['Adaptive Ridge']:.4f}, "
              f"Best Adaptive MLP={res['_best_adapt_mlp_mae']:.4f} ({res['_best_adapt_mlp']})")

    return seed_results, all_configs


def print_full_table(all_configs, title):
    """Print MAE for every activation/architecture combo."""
    print(f"\n{'='*80}")
    print(f"FULL RESULTS: {title}")
    print(f"{'='*80}")

    # Group by direct vs adaptive
    for pathway in ['Direct', 'Adaptive']:
        print(f"\n--- {pathway} pathway ---")
        print(f"{'Architecture':<25s} {'relu':>10s} {'tanh':>10s} {'logistic':>10s}")
        print("-" * 60)

        # Ridge first
        if f'{pathway} Ridge' in all_configs:
            vals = all_configs[f'{pathway} Ridge']
            print(f"{'Ridge':<25s} {np.mean(vals):>10.4f}")

        # MLPs by architecture
        for arch in ARCHITECTURES:
            row = f"{str(arch):<25s}"
            for act in ACTIVATIONS:
                key = f'{pathway} MLP {arch} {act}'
                if key in all_configs:
                    vals = all_configs[key]
                    row += f" {np.mean(vals):>10.4f}"
                else:
                    row += f" {'N/A':>10s}"
            print(row)


def plot_summary(lj_results, rose_results, savepath):
    """Bar chart comparing 4 methods for both systems."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ['Direct Ridge', 'Best Direct MLP', 'Adaptive Ridge', 'Best Adaptive MLP']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#e377c2']
    labels = ['Direct\n(Ridge)', 'Direct\n(best MLP)', 'Adaptive\n(Ridge)', 'Adaptive\n(best MLP)']

    for ax, results, title in [(axes[0], lj_results, 'LJ Extrapolation'),
                                (axes[1], rose_results, 'Rose/UBER Extrapolation')]:
        means = [np.mean(results[m]) for m in methods]
        stds = [np.std(results[m]) for m in methods]
        bars = ax.bar(range(4), means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('MAE (eV)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def plot_activation_comparison(all_configs, system_name, savepath):
    """Heatmap-style comparison: architecture x activation for direct pathway."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pathway in [(axes[0], 'Direct'), (axes[1], 'Adaptive')]:
        data = []
        arch_labels = []
        for arch in ARCHITECTURES:
            row = []
            for act in ACTIVATIONS:
                key = f'{pathway} MLP {arch} {act}'
                if key in all_configs:
                    row.append(np.mean(all_configs[key]))
                else:
                    row.append(np.nan)
            data.append(row)
            arch_labels.append(str(arch))

        data = np.array(data)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(ACTIVATIONS)))
        ax.set_xticklabels(ACTIVATIONS)
        ax.set_yticks(range(len(ARCHITECTURES)))
        ax.set_yticklabels(arch_labels, fontsize=8)
        ax.set_xlabel('Activation')
        ax.set_ylabel('Architecture')
        ax.set_title(f'{pathway} MLP ({system_name})')

        # Add values
        for i in range(len(ARCHITECTURES)):
            for j in range(len(ACTIVATIONS)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax, label='MAE (eV)')

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    figdir = 'figures'

    print("=" * 80)
    print("MLP BASELINE v2: Full activation/architecture sweep")
    print(f"Activations: {ACTIVATIONS}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Total configs per pathway: {len(ACTIVATIONS) * len(ARCHITECTURES)}")
    print("=" * 80)

    # --- LJ Extrapolation ---
    print(f"\n{'='*80}")
    print("LJ EXTRAPOLATION (N=160)")
    print(f"{'='*80}")
    lj_summary, lj_all = run_sweep('lj', 'extrap')
    print_full_table(lj_all, "LJ Extrapolation")

    # --- Rose/UBER Extrapolation ---
    print(f"\n{'='*80}")
    print("ROSE/UBER EXTRAPOLATION (N=160)")
    print(f"{'='*80}")
    rose_summary, rose_all = run_sweep('rose', 'extrap')
    print_full_table(rose_all, "Rose/UBER Extrapolation")

    # --- Plots ---
    plot_summary(lj_summary, rose_summary, f'{figdir}/fig_mlp_v2_summary.png')
    plot_activation_comparison(lj_all, 'LJ Extrap', f'{figdir}/fig_mlp_v2_lj_heatmap.png')
    plot_activation_comparison(rose_all, 'Rose Extrap', f'{figdir}/fig_mlp_v2_rose_heatmap.png')

    # --- Final summary ---
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for name, results in [('LJ', lj_summary), ('Rose/UBER', rose_summary)]:
        print(f"\n{name} Extrapolation (N=160, {len(results['Direct Ridge'])} seeds):")
        for m in ['Direct Ridge', 'Best Direct MLP', 'Adaptive Ridge', 'Best Adaptive MLP']:
            vals = results[m]
            print(f"  {m:<25s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        dr = np.mean(results['Direct Ridge'])
        dm = np.mean(results['Best Direct MLP'])
        ar = np.mean(results['Adaptive Ridge'])
        am = np.mean(results['Best Adaptive MLP'])
        print(f"  Adaptive Ridge vs Best Direct MLP: {dm/ar:.1f}x")
        print(f"  Best Adaptive MLP vs Best Direct MLP: {dm/am:.1f}x")
