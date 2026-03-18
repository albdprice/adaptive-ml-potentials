"""
LJ Learning Curves: Computation Script (for cluster)
=====================================================

Computes learning curves comparing adaptive vs direct ML approaches
for Lennard-Jones potentials using Ridge regression.

Note: KRR with RBF kernel cannot extrapolate (kernel decays to zero outside
training range). Ridge is a global linear model that can extrapolate, which
is the right choice for demonstrating the adaptive approach's advantage.

Saves results to .npz file for plotting separately.

Usage:
    python lj_learning_curves_compute.py
    python lj_learning_curves_compute.py --n-seeds 10
    python lj_learning_curves_compute.py --output my_results.npz
"""

import argparse
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# POTENTIAL FUNCTION
# =============================================================================

def lennard_jones(r, epsilon, sigma):
    """LJ potential: V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]"""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def generate_dataset(n_samples, d1_range, d2_range, n_points=50, noise_level=0.0):
    """Generate LJ potentials with descriptors on a FIXED r-grid.

    Parameters map linearly from descriptors:
        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2

    Using a fixed r-grid is critical: with a sigma-scaled grid, sigma/r is
    constant and V depends only linearly on epsilon, making the problem trivial
    for Ridge. A fixed grid means changing sigma shifts the potential well,
    creating genuinely nonlinear descriptor-to-curve dependence.
    """
    # Fixed r-grid covering range needed for all sigma values (train + test)
    # sigma ranges from ~2.75 (train) to ~4.5 (extrap test)
    # Need r > sigma to avoid the repulsive wall blowup
    r = np.linspace(2.5, 12.0, n_points)

    data = {'X': [], 'params': [], 'curves': [], 'r_grid': r, 'epsilon': []}

    for i in range(n_samples):
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)

        sigma = 2.5 + 0.5 * d1
        epsilon = 0.05 + 0.1 * d2

        V = lennard_jones(r, epsilon, sigma)

        if noise_level > 0:
            V = V + np.random.normal(0, noise_level * np.abs(V).mean(), len(V))

        data['X'].append([d1, d2])
        data['params'].append([epsilon, sigma])
        data['curves'].append(V)
        data['epsilon'].append(epsilon)

    return data


def train_and_evaluate(train_data, test_data):
    """
    Train both models using Ridge regression.

    Ridge is a global linear model that can extrapolate, making it the right
    choice for demonstrating the adaptive approach's extrapolation advantage.
    """
    X_train = np.array(train_data['X'])
    X_test = np.array(test_data['X'])
    Y_curves_train = np.array(train_data['curves'])
    Y_curves_test = np.array(test_data['curves'])
    Y_params_train = np.array(train_data['params'])
    Y_params_test = np.array(test_data['params'])

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # === DIRECT APPROACH ===
    scaler_V = StandardScaler()
    Y_train_s = scaler_V.fit_transform(Y_curves_train)

    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_s, Y_train_s)
    Y_pred_direct = scaler_V.inverse_transform(model_direct.predict(X_test_s))

    # === ADAPTIVE APPROACH ===
    scaler_params = StandardScaler()
    Y_params_train_s = scaler_params.fit_transform(Y_params_train)

    model_adaptive = Ridge(alpha=1.0)
    model_adaptive.fit(X_train_s, Y_params_train_s)
    Y_params_pred = scaler_params.inverse_transform(model_adaptive.predict(X_test_s))

    Y_pred_adaptive = []
    r = test_data['r_grid']
    for i, (eps, sig) in enumerate(Y_params_pred):
        Y_pred_adaptive.append(lennard_jones(r, max(eps, 0.001), max(sig, 0.1)))
    Y_pred_adaptive = np.array(Y_pred_adaptive)

    mse_direct = mean_squared_error(Y_curves_test, Y_pred_direct)
    mse_adaptive = mean_squared_error(Y_curves_test, Y_pred_adaptive)
    mae_direct = mean_absolute_error(Y_curves_test, Y_pred_direct)
    mae_adaptive = mean_absolute_error(Y_curves_test, Y_pred_adaptive)
    r2_direct = r2_score(Y_curves_test.flatten(), Y_pred_direct.flatten())
    r2_adaptive = r2_score(Y_curves_test.flatten(), Y_pred_adaptive.flatten())

    return {
        'mse_direct': mse_direct, 'mse_adaptive': mse_adaptive,
        'mae_direct': mae_direct, 'mae_adaptive': mae_adaptive,
        'r2_direct': r2_direct, 'r2_adaptive': r2_adaptive,
    }


def compute_learning_curves(train_sizes, n_seeds=5, test_regime='extrapolation'):
    """Compute learning curves with multiple random seeds for error bars."""
    results = {
        'train_sizes': train_sizes,
        'r2_direct_mean': [], 'r2_direct_std': [],
        'r2_adaptive_mean': [], 'r2_adaptive_std': [],
        'mse_direct_mean': [], 'mse_direct_std': [],
        'mse_adaptive_mean': [], 'mse_adaptive_std': [],
        'mae_direct_mean': [], 'mae_direct_std': [],
        'mae_adaptive_mean': [], 'mae_adaptive_std': [],
    }

    for n_train in train_sizes:
        r2_direct_runs, r2_adaptive_runs = [], []
        mse_direct_runs, mse_adaptive_runs = [], []
        mae_direct_runs, mae_adaptive_runs = [], []

        for seed in range(n_seeds):
            np.random.seed(seed * 100 + n_train)

            train_data = generate_dataset(n_train, d1_range=(0.5, 1.5), d2_range=(0.5, 1.5))

            if test_regime == 'extrapolation':
                test_data = generate_dataset(50, d1_range=(2.5, 4.0), d2_range=(0.5, 1.5))
            else:
                test_data = generate_dataset(50, d1_range=(0.7, 1.3), d2_range=(0.6, 1.4))

            if len(train_data['X']) < 5 or len(test_data['X']) < 5:
                continue

            res = train_and_evaluate(train_data, test_data)
            r2_direct_runs.append(res['r2_direct'])
            r2_adaptive_runs.append(res['r2_adaptive'])
            mse_direct_runs.append(res['mse_direct'])
            mse_adaptive_runs.append(res['mse_adaptive'])
            mae_direct_runs.append(res['mae_direct'])
            mae_adaptive_runs.append(res['mae_adaptive'])

        results['r2_direct_mean'].append(np.mean(r2_direct_runs))
        results['r2_direct_std'].append(np.std(r2_direct_runs))
        results['r2_adaptive_mean'].append(np.mean(r2_adaptive_runs))
        results['r2_adaptive_std'].append(np.std(r2_adaptive_runs))
        results['mse_direct_mean'].append(np.mean(mse_direct_runs))
        results['mse_direct_std'].append(np.std(mse_direct_runs))
        results['mse_adaptive_mean'].append(np.mean(mse_adaptive_runs))
        results['mse_adaptive_std'].append(np.std(mse_adaptive_runs))
        results['mae_direct_mean'].append(np.mean(mae_direct_runs))
        results['mae_direct_std'].append(np.std(mae_direct_runs))
        results['mae_adaptive_mean'].append(np.mean(mae_adaptive_runs))
        results['mae_adaptive_std'].append(np.std(mae_adaptive_runs))

        print(f"  N={n_train:3d}: Direct MAE={np.mean(mae_direct_runs):.6f}+/-{np.std(mae_direct_runs):.6f} eV, "
              f"Adaptive MAE={np.mean(mae_adaptive_runs):.6f}+/-{np.std(mae_adaptive_runs):.6f} eV")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute LJ learning curves')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds per training size (default: 5)')
    parser.add_argument('--output', type=str, default='lj_learning_curves_results.npz',
                        help='Output file path (default: lj_learning_curves_results.npz)')
    args = parser.parse_args()

    print("=" * 70)
    print("LJ LEARNING CURVES: Computation")
    print("Adaptive vs Direct Learning for Lennard-Jones Potentials")
    print("=" * 70)
    print(f"ML Method: Ridge Regression (global linear model)")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {args.output}")

    train_sizes = [10, 15, 20, 30, 40, 60, 80, 100, 120, 160]

    t0 = time.time()

    print(f"\n--- EXTRAPOLATION REGIME ---")
    print(f"Training: sigma ~ 2.75-3.25 A | Testing: sigma ~ 3.75-4.5 A")
    results_extrap = compute_learning_curves(
        train_sizes, n_seeds=args.n_seeds, test_regime='extrapolation')

    print(f"\n--- INTERPOLATION REGIME ---")
    print(f"Training: sigma ~ 2.75-3.25 A | Testing: within range")
    results_interp = compute_learning_curves(
        train_sizes, n_seeds=args.n_seeds, test_regime='interpolation')

    elapsed = time.time() - t0
    print(f"\nComputation finished in {elapsed:.1f}s")

    np.savez(args.output,
             train_sizes=np.array(train_sizes),
             extrap_r2_direct_mean=np.array(results_extrap['r2_direct_mean']),
             extrap_r2_direct_std=np.array(results_extrap['r2_direct_std']),
             extrap_r2_adaptive_mean=np.array(results_extrap['r2_adaptive_mean']),
             extrap_r2_adaptive_std=np.array(results_extrap['r2_adaptive_std']),
             extrap_mse_direct_mean=np.array(results_extrap['mse_direct_mean']),
             extrap_mse_direct_std=np.array(results_extrap['mse_direct_std']),
             extrap_mse_adaptive_mean=np.array(results_extrap['mse_adaptive_mean']),
             extrap_mse_adaptive_std=np.array(results_extrap['mse_adaptive_std']),
             extrap_mae_direct_mean=np.array(results_extrap['mae_direct_mean']),
             extrap_mae_direct_std=np.array(results_extrap['mae_direct_std']),
             extrap_mae_adaptive_mean=np.array(results_extrap['mae_adaptive_mean']),
             extrap_mae_adaptive_std=np.array(results_extrap['mae_adaptive_std']),
             interp_r2_direct_mean=np.array(results_interp['r2_direct_mean']),
             interp_r2_direct_std=np.array(results_interp['r2_direct_std']),
             interp_r2_adaptive_mean=np.array(results_interp['r2_adaptive_mean']),
             interp_r2_adaptive_std=np.array(results_interp['r2_adaptive_std']),
             interp_mse_direct_mean=np.array(results_interp['mse_direct_mean']),
             interp_mse_direct_std=np.array(results_interp['mse_direct_std']),
             interp_mse_adaptive_mean=np.array(results_interp['mse_adaptive_mean']),
             interp_mse_adaptive_std=np.array(results_interp['mse_adaptive_std']),
             interp_mae_direct_mean=np.array(results_interp['mae_direct_mean']),
             interp_mae_direct_std=np.array(results_interp['mae_direct_std']),
             interp_mae_adaptive_mean=np.array(results_interp['mae_adaptive_mean']),
             interp_mae_adaptive_std=np.array(results_interp['mae_adaptive_std']),
             n_seeds=np.array(args.n_seeds),
             )

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
