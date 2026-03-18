"""
LJ Learning Curves: Plotting Script
=====================================

Loads results from lj_learning_curves_compute.py (.npz file) and generates
publication-quality figures.

Usage:
    python lj_learning_curves_plot.py
    python lj_learning_curves_plot.py --input my_results.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
})


def power_law(x, a, b):
    return a * np.power(x, -b)


def fit_power_law(x, y):
    try:
        log_x = np.log(x)
        log_y = np.log(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        b = -coeffs[0]
        a = np.exp(coeffs[1])
        return a, b
    except Exception:
        return None, None


def load_results(path):
    data = np.load(path)
    train_sizes = data['train_sizes']

    results_extrap = {
        'train_sizes': train_sizes.tolist(),
        'r2_direct_mean': data['extrap_r2_direct_mean'],
        'r2_direct_std': data['extrap_r2_direct_std'],
        'r2_adaptive_mean': data['extrap_r2_adaptive_mean'],
        'r2_adaptive_std': data['extrap_r2_adaptive_std'],
        'mse_direct_mean': data['extrap_mse_direct_mean'],
        'mse_direct_std': data['extrap_mse_direct_std'],
        'mse_adaptive_mean': data['extrap_mse_adaptive_mean'],
        'mse_adaptive_std': data['extrap_mse_adaptive_std'],
        'mae_direct_mean': data['extrap_mae_direct_mean'],
        'mae_direct_std': data['extrap_mae_direct_std'],
        'mae_adaptive_mean': data['extrap_mae_adaptive_mean'],
        'mae_adaptive_std': data['extrap_mae_adaptive_std'],
    }

    results_interp = {
        'train_sizes': train_sizes.tolist(),
        'r2_direct_mean': data['interp_r2_direct_mean'],
        'r2_direct_std': data['interp_r2_direct_std'],
        'r2_adaptive_mean': data['interp_r2_adaptive_mean'],
        'r2_adaptive_std': data['interp_r2_adaptive_std'],
        'mse_direct_mean': data['interp_mse_direct_mean'],
        'mse_direct_std': data['interp_mse_direct_std'],
        'mse_adaptive_mean': data['interp_mse_adaptive_mean'],
        'mse_adaptive_std': data['interp_mse_adaptive_std'],
        'mae_direct_mean': data['interp_mae_direct_mean'],
        'mae_direct_std': data['interp_mae_direct_std'],
        'mae_adaptive_mean': data['interp_mae_adaptive_mean'],
        'mae_adaptive_std': data['interp_mae_adaptive_std'],
    }

    n_seeds = int(data['n_seeds'])
    return results_extrap, results_interp, n_seeds


def plot_learning_curves(results_extrap, results_interp, save_path='fig_lj_learning_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_sizes = np.array(results_extrap['train_sizes'])
    color_direct = '#1f77b4'
    color_adaptive = '#d62728'

    # --- Panel A: Extrapolation ---
    ax = axes[0]
    mae_direct = np.array(results_extrap['mae_direct_mean'])
    mae_adaptive = np.array(results_extrap['mae_adaptive_mean'])

    ax.errorbar(train_sizes, mae_direct, yerr=results_extrap['mae_direct_std'],
                fmt='o', color=color_direct, linewidth=2, markersize=10,
                capsize=4, label='Direct (learn V)', alpha=0.9)
    ax.errorbar(train_sizes, mae_adaptive, yerr=results_extrap['mae_adaptive_std'],
                fmt='s', color=color_adaptive, linewidth=2, markersize=10,
                capsize=4, label='Adaptive (learn params)', alpha=0.9)

    x_fit = np.linspace(train_sizes.min(), train_sizes.max(), 100)
    a_d, b_d = fit_power_law(train_sizes, mae_direct)
    if a_d is not None:
        ax.plot(x_fit, power_law(x_fit, a_d, b_d), '--', color=color_direct,
                linewidth=2, alpha=0.7, label=f'Direct fit: N$^{{-{b_d:.2f}}}$')

    mask = train_sizes >= 20
    a_a, b_a = fit_power_law(train_sizes[mask], mae_adaptive[mask])
    if a_a is not None:
        ax.plot(x_fit, power_law(x_fit, a_a, b_a), '--', color=color_adaptive,
                linewidth=2, alpha=0.7, label=f'Adaptive fit: N$^{{-{b_a:.2f}}}$')

    ax.set_xlabel('Training Set Size (N)', fontsize=13)
    ax.set_ylabel('MAE [eV]', fontsize=13)
    ax.set_title('A. Learning Curves (Extrapolation)\nTest: sigma ~ 3.75-4.5 A', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel B: Interpolation ---
    ax = axes[1]
    mae_direct_i = np.array(results_interp['mae_direct_mean'])
    mae_adaptive_i = np.array(results_interp['mae_adaptive_mean'])

    ax.errorbar(train_sizes, mae_direct_i, yerr=results_interp['mae_direct_std'],
                fmt='o', color=color_direct, linewidth=2, markersize=10,
                capsize=4, label='Direct', alpha=0.9)
    ax.errorbar(train_sizes, mae_adaptive_i, yerr=results_interp['mae_adaptive_std'],
                fmt='s', color=color_adaptive, linewidth=2, markersize=10,
                capsize=4, label='Adaptive', alpha=0.9)

    a_d_i, b_d_i = fit_power_law(train_sizes, mae_direct_i)
    if a_d_i is not None:
        ax.plot(x_fit, power_law(x_fit, a_d_i, b_d_i), '--', color=color_direct,
                linewidth=2, alpha=0.7, label=f'Direct: N$^{{-{b_d_i:.2f}}}$')

    a_a_i, b_a_i = fit_power_law(train_sizes, mae_adaptive_i)
    if a_a_i is not None:
        ax.plot(x_fit, power_law(x_fit, a_a_i, b_a_i), '--', color=color_adaptive,
                linewidth=2, alpha=0.7, label=f'Adaptive: N$^{{-{b_a_i:.2f}}}$')

    ax.set_xlabel('Training Set Size (N)', fontsize=13)
    ax.set_ylabel('MAE [eV]', fontsize=13)
    ax.set_title('B. Learning Curves (Interpolation)\nTest within training range', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved: {save_path}")


def create_combined_figure(results_extrap, results_interp, save_path='fig_lj_learning_curves_combined.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    train_sizes = np.array(results_extrap['train_sizes'])
    color_direct = '#1f77b4'
    color_adaptive = '#d62728'

    mae_direct = np.array(results_extrap['mae_direct_mean'])
    mae_adaptive = np.array(results_extrap['mae_adaptive_mean'])
    mae_direct_i = np.array(results_interp['mae_direct_mean'])
    mae_adaptive_i = np.array(results_interp['mae_adaptive_mean'])

    x_fit = np.linspace(train_sizes.min(), train_sizes.max(), 100)

    # --- Panel A: Extrapolation ---
    ax = axes[0]
    ax.errorbar(train_sizes, mae_direct, yerr=results_extrap['mae_direct_std'],
                fmt='o', color=color_direct, linewidth=2.5, markersize=10,
                capsize=5, label='Direct (learn V)', alpha=0.9)
    ax.errorbar(train_sizes, mae_adaptive, yerr=results_extrap['mae_adaptive_std'],
                fmt='s', color=color_adaptive, linewidth=2.5, markersize=10,
                capsize=5, label='Adaptive (learn params)', alpha=0.9)

    a_d, b_d = fit_power_law(train_sizes, mae_direct)
    if a_d is not None:
        ax.plot(x_fit, power_law(x_fit, a_d, b_d), '--', color=color_direct,
                linewidth=2, alpha=0.7, label=f'N$^{{-{b_d:.2f}}}$')

    mask = train_sizes >= 20
    a_a, b_a = fit_power_law(train_sizes[mask], mae_adaptive[mask])
    if a_a is not None:
        ax.plot(x_fit, power_law(x_fit, a_a, b_a), '--', color=color_adaptive,
                linewidth=2, alpha=0.7, label=f'N$^{{-{b_a:.2f}}}$')

    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('MAE [eV]')
    ax.set_title('A. Extrapolation\n(Test outside training range)')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel B: Interpolation ---
    ax = axes[1]
    ax.errorbar(train_sizes, mae_direct_i, yerr=results_interp['mae_direct_std'],
                fmt='o', color=color_direct, linewidth=2.5, markersize=10,
                capsize=5, label='Direct', alpha=0.9)
    ax.errorbar(train_sizes, mae_adaptive_i, yerr=results_interp['mae_adaptive_std'],
                fmt='s', color=color_adaptive, linewidth=2.5, markersize=10,
                capsize=5, label='Adaptive', alpha=0.9)

    a_d_i, b_d_i = fit_power_law(train_sizes, mae_direct_i)
    if a_d_i is not None:
        ax.plot(x_fit, power_law(x_fit, a_d_i, b_d_i), '--', color=color_direct,
                linewidth=2, alpha=0.7, label=f'N$^{{-{b_d_i:.2f}}}$')

    a_a_i, b_a_i = fit_power_law(train_sizes, mae_adaptive_i)
    if a_a_i is not None:
        ax.plot(x_fit, power_law(x_fit, a_a_i, b_a_i), '--', color=color_adaptive,
                linewidth=2, alpha=0.7, label=f'N$^{{-{b_a_i:.2f}}}$')

    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('MAE [eV]')
    ax.set_title('B. Interpolation\n(Test within training range)')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel C: Improvement Factor ---
    ax = axes[2]
    improvement_extrap = mae_direct / mae_adaptive
    improvement_interp = mae_direct_i / mae_adaptive_i

    ax.plot(train_sizes, improvement_extrap, 'o-', color='#2ca02c', linewidth=2.5,
            markersize=10, label='Extrapolation')
    ax.plot(train_sizes, improvement_interp, 's--', color='#ff7f0e', linewidth=2.5,
            markersize=10, label='Interpolation')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Equal performance')
    ax.fill_between(train_sizes, 1, improvement_extrap,
                    where=(improvement_extrap > 1), alpha=0.2, color='#2ca02c')

    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('MAE Ratio (Direct / Adaptive)')
    ax.set_title('C. Adaptive Improvement Factor\n(>1 means adaptive is better)')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    valid_idx = np.where(improvement_extrap > 1)[0]
    if len(valid_idx) > 0:
        max_idx = valid_idx[np.argmax(improvement_extrap[valid_idx])]
        ax.annotate(f'{improvement_extrap[max_idx]:.1f}x better',
                   xy=(train_sizes[max_idx], improvement_extrap[max_idx]),
                   xytext=(train_sizes[max_idx] * 1.3, improvement_extrap[max_idx] * 1.3),
                   fontsize=11, fontweight='bold', color='#2ca02c',
                   arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved: {save_path}")


def print_summary_table(results_extrap, results_interp):
    print("\n" + "=" * 100)
    print("LJ LEARNING CURVES SUMMARY TABLE (MAE in eV)")
    print("ML Method: Ridge Regression (global linear model)")
    print("=" * 100)

    train_sizes = results_extrap['train_sizes']
    train_arr = np.array(train_sizes)

    print("\n--- EXTRAPOLATION REGIME (Test: sigma ~ 3.75-4.5 A, Train: sigma ~ 2.75-3.25 A) ---")
    print(f"{'N_train':<10} {'Direct MAE [eV]':<22} {'Adaptive MAE [eV]':<22} {'Ratio (D/A)':<12}")
    print("-" * 80)

    mae_d_arr = np.array(results_extrap['mae_direct_mean'])
    mae_a_arr = np.array(results_extrap['mae_adaptive_mean'])

    for i, n in enumerate(train_sizes):
        mae_d = results_extrap['mae_direct_mean'][i]
        mae_d_std = results_extrap['mae_direct_std'][i]
        mae_a = results_extrap['mae_adaptive_mean'][i]
        mae_a_std = results_extrap['mae_adaptive_std'][i]
        ratio = mae_d / mae_a if mae_a > 0 else 0
        print(f"{n:<10} {mae_d:.6f} +/- {mae_d_std:.6f}  {mae_a:.6f} +/- {mae_a_std:.6f}  {ratio:.2f}x")

    a_d, b_d = fit_power_law(train_arr, mae_d_arr)
    mask = train_arr >= 20
    a_a, b_a = fit_power_law(train_arr[mask], mae_a_arr[mask])

    print("-" * 80)
    if a_d is not None:
        print(f"Direct power-law fit: MAE ~ N^(-{b_d:.2f})")
    if a_a is not None:
        print(f"Adaptive power-law fit: MAE ~ N^(-{b_a:.2f})  (fitted for N>=20)")

    print("\n--- INTERPOLATION REGIME (Test within training range) ---")
    print(f"{'N_train':<10} {'Direct MAE [eV]':<22} {'Adaptive MAE [eV]':<22} {'Ratio (D/A)':<12}")
    print("-" * 80)

    mae_d_i = np.array(results_interp['mae_direct_mean'])
    mae_a_i = np.array(results_interp['mae_adaptive_mean'])

    for i, n in enumerate(train_sizes):
        mae_d = results_interp['mae_direct_mean'][i]
        mae_d_std = results_interp['mae_direct_std'][i]
        mae_a = results_interp['mae_adaptive_mean'][i]
        mae_a_std = results_interp['mae_adaptive_std'][i]
        ratio = mae_d / mae_a if mae_a > 0 else 0
        print(f"{n:<10} {mae_d:.6f} +/- {mae_d_std:.6f}  {mae_a:.6f} +/- {mae_a_std:.6f}  {ratio:.2f}x")

    a_d_i, b_d_i = fit_power_law(train_arr, mae_d_i)
    a_a_i, b_a_i = fit_power_law(train_arr, mae_a_i)

    print("-" * 80)
    if a_d_i is not None:
        print(f"Direct power-law fit: MAE ~ N^(-{b_d_i:.2f})")
    if a_a_i is not None:
        print(f"Adaptive power-law fit: MAE ~ N^(-{b_a_i:.2f})")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Plot LJ learning curve results')
    parser.add_argument('--input', type=str, default='lj_learning_curves_results.npz',
                        help='Input .npz file (default: lj_learning_curves_results.npz)')
    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    results_extrap, results_interp, n_seeds = load_results(args.input)
    print(f"Loaded: {len(results_extrap['train_sizes'])} training sizes, {n_seeds} seeds")

    plot_learning_curves(results_extrap, results_interp)
    create_combined_figure(results_extrap, results_interp)
    print_summary_table(results_extrap, results_interp)


if __name__ == "__main__":
    main()
