"""
MLP nonlinear baseline comparison v3: Full activation sweep including SiLU.

Tests sklearn activations (relu, tanh, logistic/sigmoid) plus PyTorch SiLU.
Tests both Direct and Adaptive pathways.
7 architectures x 4 activations x 2 pathways = 56 MLP variants.

Usage:
    MPLBACKEND=Agg python mlp_baseline_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Potentials
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
# PyTorch MLP for SiLU
# =============================================================================

class TorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='silu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh,
                  'sigmoid': nn.Sigmoid}[activation]
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch_mlp(X_train, Y_train, X_test, hidden_layers, activation,
                     lr=0.001, weight_decay=0.001, max_epochs=3000, patience=20):
    """Train a PyTorch MLP with early stopping."""
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # Standardize
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    Y_mean, Y_std = Y_train.mean(0), Y_train.std(0) + 1e-8
    X_tr_s = (X_train - X_mean) / X_std
    Y_tr_s = (Y_train - Y_mean) / Y_std

    # Validation split
    n = len(X_tr_s)
    n_val = max(2, int(0.15 * n))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_t = torch.FloatTensor(X_tr_s[train_idx])
    Y_t = torch.FloatTensor(Y_tr_s[train_idx])
    X_v = torch.FloatTensor(X_tr_s[val_idx])
    Y_v = torch.FloatTensor(Y_tr_s[val_idx])

    model = TorchMLP(input_dim, output_dim, hidden_layers, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), Y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Predict
    X_te_s = (X_test - X_mean) / X_std
    with torch.no_grad():
        Y_pred_s = model(torch.FloatTensor(X_te_s)).numpy()
    Y_pred = Y_pred_s * Y_std + Y_mean
    return Y_pred

# =============================================================================
# sklearn MLP wrapper
# =============================================================================

SKLEARN_ACTIVATIONS = ['relu', 'tanh', 'logistic']
ARCHITECTURES = [
    (32, 32),
    (64, 64),
    (128, 64),
    (256, 128),
    (64, 64, 64),
    (128, 64, 32),
    (128, 128, 64, 32),
]

def train_sklearn_mlp(X_train, Y_train, X_test, hidden_layers, activation):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    Y_tr = scaler_Y.fit_transform(Y_train)
    X_te = scaler_X.transform(X_test)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers, activation=activation,
        max_iter=3000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, learning_rate='adaptive',
        learning_rate_init=0.001, alpha=0.001, random_state=42,
    )
    mlp.fit(X_tr, Y_tr)
    return scaler_Y.inverse_transform(mlp.predict(X_te))

# =============================================================================
# Full evaluation
# =============================================================================

def evaluate_all(system, train_data, test_data):
    """Run all model variants and return results dict."""
    X_train, X_test = train_data['X'], test_data['X']
    Y_test = test_data['curves']

    if system == 'lj':
        Y_train = train_data['curves']
        P_train = train_data['params']
        r = train_data['r']
        def reconstruct(params):
            return np.array([lennard_jones(r, max(e, 0.001), max(s, 0.1))
                             for e, s in params])
    else:
        Y_train = train_data['curves']
        P_train = train_data['rose_params']
        def reconstruct(params):
            curves = []
            for i, (E_c, r_e, l) in enumerate(params):
                r_i = test_data['r_grids'][i]
                curves.append(rose_V(r_i, max(E_c, 0.01), max(r_e, 0.5), max(l, 0.01)))
            return np.array(curves)

    results = {}

    # --- Ridge baselines ---
    # Direct Ridge
    scaler_X = StandardScaler()
    scaler_V = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)
    Y_tr_s = scaler_V.fit_transform(Y_train)
    m = Ridge(alpha=1.0).fit(X_tr, Y_tr_s)
    Y_pred = scaler_V.inverse_transform(m.predict(X_te))
    results['Direct Ridge'] = mean_absolute_error(Y_test, Y_pred)

    # Adaptive Ridge
    scaler_P = StandardScaler()
    P_tr_s = scaler_P.fit_transform(P_train)
    m = Ridge(alpha=1.0).fit(X_tr, P_tr_s)
    P_pred = scaler_P.inverse_transform(m.predict(X_te))
    Y_pred = reconstruct(P_pred)
    results['Adaptive Ridge'] = mean_absolute_error(Y_test, Y_pred)

    # --- sklearn MLPs (relu, tanh, logistic/sigmoid) ---
    for pathway, train_targets in [('Direct', Y_train), ('Adaptive', P_train)]:
        for arch in ARCHITECTURES:
            for act in SKLEARN_ACTIVATIONS:
                label = act if act != 'logistic' else 'sigmoid'
                name = f'{pathway} MLP {arch} {label}'
                Y_pred_raw = train_sklearn_mlp(X_train, train_targets, X_test, arch, act)
                if pathway == 'Adaptive':
                    Y_pred = reconstruct(Y_pred_raw)
                else:
                    Y_pred = Y_pred_raw
                results[name] = mean_absolute_error(Y_test, Y_pred)

    # --- PyTorch MLPs (SiLU) ---
    for pathway, train_targets in [('Direct', Y_train), ('Adaptive', P_train)]:
        for arch in ARCHITECTURES:
            name = f'{pathway} MLP {arch} silu'
            np.random.seed(42)
            torch.manual_seed(42)
            Y_pred_raw = train_torch_mlp(X_train, train_targets, X_test, arch, 'silu')
            if pathway == 'Adaptive':
                Y_pred = reconstruct(Y_pred_raw)
            else:
                Y_pred = Y_pred_raw
            results[name] = mean_absolute_error(Y_test, Y_pred)

    return results


def summarize_results(results):
    """Extract best direct/adaptive MLP from full results."""
    best_direct = ('', np.inf)
    best_adaptive = ('', np.inf)
    for k, v in results.items():
        if k.startswith('Direct MLP') and v < best_direct[1]:
            best_direct = (k, v)
        if k.startswith('Adaptive MLP') and v < best_adaptive[1]:
            best_adaptive = (k, v)
    return {
        'Direct Ridge': results['Direct Ridge'],
        'Best Direct MLP': best_direct[1],
        'Best Direct MLP name': best_direct[0],
        'Adaptive Ridge': results['Adaptive Ridge'],
        'Best Adaptive MLP': best_adaptive[1],
        'Best Adaptive MLP name': best_adaptive[0],
    }


# =============================================================================
# Printing
# =============================================================================

def print_table(all_results, title):
    """Print full activation x architecture table."""
    print(f"\n{'='*90}")
    print(f"{title}")
    print(f"{'='*90}")

    activations = ['relu', 'tanh', 'sigmoid', 'silu']
    for pathway in ['Direct', 'Adaptive']:
        print(f"\n--- {pathway} pathway ---")
        print(f"{'Architecture':<25s} {'relu':>10s} {'tanh':>10s} {'sigmoid':>10s} {'silu':>10s}")
        print("-" * 70)

        if f'{pathway} Ridge' in all_results:
            vals = all_results[f'{pathway} Ridge']
            print(f"{'Ridge':<25s} {np.mean(vals):>10.4f}")

        for arch in ARCHITECTURES:
            row = f"{str(arch):<25s}"
            for act in activations:
                key = f'{pathway} MLP {arch} {act}'
                if key in all_results:
                    vals = all_results[key]
                    row += f" {np.mean(vals):>10.4f}"
                else:
                    row += f" {'N/A':>10s}"
            print(row)


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmaps(all_results, system_name, savepath):
    """Heatmap: architecture x activation for direct and adaptive."""
    activations = ['relu', 'tanh', 'sigmoid', 'silu']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, pathway in [(axes[0], 'Direct'), (axes[1], 'Adaptive')]:
        data = []
        for arch in ARCHITECTURES:
            row = []
            for act in activations:
                key = f'{pathway} MLP {arch} {act}'
                if key in all_results:
                    row.append(np.mean(all_results[key]))
                else:
                    row.append(np.nan)
            data.append(row)

        data = np.array(data)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(activations)
        ax.set_yticks(range(len(ARCHITECTURES)))
        ax.set_yticklabels([str(a) for a in ARCHITECTURES], fontsize=8)
        ax.set_xlabel('Activation')
        ax.set_ylabel('Architecture')
        ax.set_title(f'{pathway} MLP ({system_name})')

        for i in range(len(ARCHITECTURES)):
            for j in range(len(activations)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                            fontsize=8, color='white' if data[i,j] > np.nanmedian(data) else 'black')

        plt.colorbar(im, ax=ax, label='MAE (eV)')

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def plot_summary_bar(lj_summaries, rose_summaries, savepath):
    """Bar chart: 4 methods for both systems."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = ['Direct Ridge', 'Best Direct MLP', 'Adaptive Ridge', 'Best Adaptive MLP']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#e377c2']
    labels = ['Direct\n(Ridge)', 'Direct\n(best MLP)', 'Adaptive\n(Ridge)', 'Adaptive\n(best MLP)']

    for ax, summaries, title in [(axes[0], lj_summaries, 'LJ Extrapolation'),
                                  (axes[1], rose_summaries, 'Rose/UBER Extrapolation')]:
        means = [np.mean([s[m] for s in summaries]) for m in methods]
        stds = [np.std([s[m] for s in summaries]) for m in methods]
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


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    figdir = 'figures'
    n_seeds = 5
    n_train = 160
    activations_tested = ['relu', 'tanh', 'sigmoid', 'silu']

    print("=" * 90)
    print("MLP BASELINE v3: Full sweep with SiLU")
    print(f"Activations: {activations_tested}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Configs per pathway: {len(activations_tested) * len(ARCHITECTURES)} MLP + 1 Ridge")
    print(f"Total: 2 pathways x ({len(activations_tested)*len(ARCHITECTURES)} MLP + 1 Ridge) = "
          f"{2*(len(activations_tested)*len(ARCHITECTURES)+1)} models")
    print("=" * 90)

    # Collect all results across seeds
    lj_all = {}
    rose_all = {}
    lj_summaries = []
    rose_summaries = []

    for s in range(n_seeds):
        seed = s * 100 + n_train
        print(f"\n--- Seed {s} ---")

        # LJ
        train_lj = generate_lj_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
        test_lj = generate_lj_dataset(50, (2.5, 4.0), (0.5, 1.5), seed=seed + 10000)
        res_lj = evaluate_all('lj', train_lj, test_lj)
        summary_lj = summarize_results(res_lj)
        lj_summaries.append(summary_lj)

        print(f"  LJ:   Direct Ridge={res_lj['Direct Ridge']:.4f}, "
              f"Best Direct MLP={summary_lj['Best Direct MLP']:.4f} ({summary_lj['Best Direct MLP name']}), "
              f"Adaptive Ridge={res_lj['Adaptive Ridge']:.4f}")

        for k, v in res_lj.items():
            if isinstance(v, float):
                lj_all.setdefault(k, []).append(v)

        # Rose
        train_rose = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
        test_rose = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
        res_rose = evaluate_all('rose', train_rose, test_rose)
        summary_rose = summarize_results(res_rose)
        rose_summaries.append(summary_rose)

        print(f"  Rose: Direct Ridge={res_rose['Direct Ridge']:.4f}, "
              f"Best Direct MLP={summary_rose['Best Direct MLP']:.4f} ({summary_rose['Best Direct MLP name']}), "
              f"Adaptive Ridge={res_rose['Adaptive Ridge']:.4f}")

        for k, v in res_rose.items():
            if isinstance(v, float):
                rose_all.setdefault(k, []).append(v)

    # --- Full tables ---
    print_table(lj_all, "LJ EXTRAPOLATION (N=160)")
    print_table(rose_all, "ROSE/UBER EXTRAPOLATION (N=160)")

    # --- Plots ---
    plot_heatmaps(lj_all, 'LJ Extrap', f'{figdir}/fig_mlp_v3_lj_heatmap.png')
    plot_heatmaps(rose_all, 'Rose Extrap', f'{figdir}/fig_mlp_v3_rose_heatmap.png')
    plot_summary_bar(lj_summaries, rose_summaries, f'{figdir}/fig_mlp_v3_summary.png')

    # --- Final summary ---
    print(f"\n{'='*90}")
    print("FINAL SUMMARY")
    print(f"{'='*90}")
    for name, summaries, all_res in [('LJ', lj_summaries, lj_all),
                                      ('Rose/UBER', rose_summaries, rose_all)]:
        print(f"\n{name} Extrapolation (N={n_train}, {n_seeds} seeds):")
        for m in ['Direct Ridge', 'Best Direct MLP', 'Adaptive Ridge', 'Best Adaptive MLP']:
            vals = [s[m] for s in summaries]
            print(f"  {m:<25s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            if m == 'Best Direct MLP':
                names = [s['Best Direct MLP name'] for s in summaries]
                from collections import Counter
                print(f"    Best config: {Counter(names).most_common(1)[0][0]}")
            if m == 'Best Adaptive MLP':
                names = [s['Best Adaptive MLP name'] for s in summaries]
                from collections import Counter
                print(f"    Best config: {Counter(names).most_common(1)[0][0]}")

        dr = np.mean([s['Direct Ridge'] for s in summaries])
        dm = np.mean([s['Best Direct MLP'] for s in summaries])
        ar = np.mean([s['Adaptive Ridge'] for s in summaries])
        am = np.mean([s['Best Adaptive MLP'] for s in summaries])
        print(f"\n  Ratios:")
        print(f"    Adaptive Ridge vs Direct Ridge:    {dr/ar:.1f}x")
        print(f"    Adaptive Ridge vs Best Direct MLP: {dm/ar:.1f}x")
        print(f"    Best Adaptive MLP vs Best Direct MLP: {dm/am:.1f}x")
