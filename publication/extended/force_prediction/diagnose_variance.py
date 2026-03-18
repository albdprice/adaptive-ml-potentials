"""Diagnose large error bars in Rose/UBER force learning curves."""
import numpy as np
from force_learning_curves import *

train_sizes = [10, 15, 20, 30, 40, 60, 80, 100, 120, 160]
n_seeds = 5

for regime in ['extrap', 'interp']:
    print(f"\n{'='*60}")
    print(f"Rose/UBER {'Extrapolation' if regime=='extrap' else 'Interpolation'}")
    print(f"{'='*60}")
    print(f"{'N':>4s}  {'F_adapt mean':>12s} {'F_adapt std':>12s}  {'F_direct mean':>13s} {'F_direct std':>12s}  per-seed F_adaptive")

    for n_train in train_sizes:
        fd_vals, fa_vals = [], []
        for s in range(n_seeds):
            seed = s * 100 + n_train
            train = generate_rose_dataset(n_train, (0.5, 1.5), (0.5, 1.5), seed=seed)
            if regime == 'extrap':
                test = generate_rose_dataset(50, (3.0, 4.5), (0.5, 1.5), seed=seed + 10000)
            else:
                test = generate_rose_dataset(50, (0.7, 1.3), (0.6, 1.4), seed=seed + 10000)
            res = evaluate_forces_rose(train, test)
            fd_vals.append(res['mae_F_direct'])
            fa_vals.append(res['mae_F_adaptive'])

        fa_arr = np.array(fa_vals)
        fd_arr = np.array(fd_vals)
        seeds_str = ', '.join(f'{v:.4f}' for v in fa_vals)
        print(f"{n_train:4d}  {fa_arr.mean():12.4f} {fa_arr.std():12.4f}  {fd_arr.mean():13.4f} {fd_arr.std():12.4f}  [{seeds_str}]")
