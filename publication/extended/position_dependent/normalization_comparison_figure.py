"""
Two clean figures for Anatole:
  1. E_elec unnormalized — train vs test ranges are wildly different
  2. E_elec / Z² normalized — train vs test overlap perfectly

Shows why Z² normalization makes the problem universal.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ================================================================
# Dataset
# ================================================================
def make_molecules(Z_values):
    r = np.linspace(1.0, 8.0, 200)  # dense grid for smooth curves
    Z = np.array(Z_values, dtype=float)
    Z_AB = Z**2
    D_e = 0.1 + 0.025 * (Z - 3)
    R_e = np.full(len(Z), 2.2)
    alpha = np.full(len(Z), 0.9)

    V_total = np.zeros((len(Z), len(r)))
    V_nn = np.zeros((len(Z), len(r)))
    E_elec = np.zeros((len(Z), len(r)))

    for i in range(len(Z)):
        u = np.exp(-alpha[i] * (r - R_e[i]))
        V_total[i] = D_e[i] * (1 - u)**2 - D_e[i]
        V_nn[i] = Z_AB[i] / r
        E_elec[i] = V_total[i] - V_nn[i]

    return {'Z': Z, 'Z_AB': Z_AB, 'r': r,
            'V_total': V_total, 'V_nn': V_nn, 'E_elec': E_elec}


Z_train = list(range(5, 14))
Z_test = list(range(17, 28))

train = make_molecules(Z_train)
test = make_molecules(Z_test)
r = train['r']

# ================================================================
# FIGURE 1: Unnormalized E_elec
# ================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

for i in range(len(Z_train)):
    Z_i = train['Z'][i]
    ax1.plot(r, train['E_elec'][i], color='#1f77b4', lw=1.5, alpha=0.6,
             label=f'Z={Z_i:.0f}' if i == 0 or i == len(Z_train)-1 else None)

for i in range(len(Z_test)):
    Z_i = test['Z'][i]
    ax1.plot(r, test['E_elec'][i], color='#d62728', lw=1.5, alpha=0.6,
             label=f'Z={Z_i:.0f}' if i == 0 or i == len(Z_test)-1 else None)

# Range annotations
E_tr_min, E_tr_max = train['E_elec'].min(), train['E_elec'].max()
E_te_min, E_te_max = test['E_elec'].min(), test['E_elec'].max()

ax1.annotate('', xy=(7.5, E_tr_min), xytext=(7.5, E_tr_max),
            arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2))
ax1.text(7.7, (E_tr_min + E_tr_max)/2,
         f'Train range\n[{E_tr_min:.0f}, {E_tr_max:.1f}]',
         fontsize=9, color='#1f77b4', va='center')

ax1.annotate('', xy=(6.5, E_te_min), xytext=(6.5, E_te_max),
            arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2))
ax1.text(5.0, E_te_min * 0.6,
         f'Test range\n[{E_te_min:.0f}, {E_te_max:.1f}]',
         fontsize=9, color='#d62728', va='center')

ax1.set_xlabel('r [Bohr]', fontsize=12)
ax1.set_ylabel('$E_{elec}$ [Ha]', fontsize=12)
ax1.set_title('$E_{elec}(r)$ — unnormalized\nTrain (blue, Z=5-13) vs Test (red, Z=17-27)', fontsize=13)
ax1.legend(fontsize=9, ncol=2, loc='lower right')
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'fig_Eelec_unnormalized.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_Eelec_unnormalized.png')
plt.close(fig1)

# ================================================================
# FIGURE 2: Normalized E_elec / Z²
# ================================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

E_norm_train = train['E_elec'] / train['Z_AB'][:, np.newaxis]
E_norm_test = test['E_elec'] / test['Z_AB'][:, np.newaxis]

for i in range(len(Z_train)):
    Z_i = train['Z'][i]
    ax2.plot(r, E_norm_train[i], color='#1f77b4', lw=1.5, alpha=0.6,
             label=f'Z={Z_i:.0f}' if i == 0 or i == len(Z_train)-1 else None)

for i in range(len(Z_test)):
    Z_i = test['Z'][i]
    ax2.plot(r, E_norm_test[i], color='#d62728', lw=1.5, alpha=0.6,
             label=f'Z={Z_i:.0f}' if i == 0 or i == len(Z_test)-1 else None)

# Universal -1/r reference
ax2.plot(r, -1.0/r, 'k--', lw=2, alpha=0.5, label='$-1/r$ (universal)')

# Range annotations
En_tr_min, En_tr_max = E_norm_train.min(), E_norm_train.max()
En_te_min, En_te_max = E_norm_test.min(), E_norm_test.max()

ax2.text(5.5, -0.35,
         f'Train range: [{En_tr_min:.3f}, {En_tr_max:.3f}]\n'
         f'Test range:  [{En_te_min:.3f}, {En_te_max:.3f}]\n'
         f'Nearly identical!',
         fontsize=10, va='center',
         bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.9))

ax2.set_xlabel('r [Bohr]', fontsize=12)
ax2.set_ylabel('$E_{elec} / Z^2$ [Ha]', fontsize=12)
ax2.set_title('$E_{elec}(r) / Z^2$ — normalized by nuclear charge\nTrain (blue) and Test (red) overlap', fontsize=13)
ax2.legend(fontsize=9, ncol=2, loc='lower right')
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'fig_Eelec_normalized.png'),
            dpi=150, bbox_inches='tight')
print('Saved fig_Eelec_normalized.png')
plt.close(fig2)

# ================================================================
# Stats
# ================================================================
print(f'\n{"="*60}')
print(f'STATISTICS')
print(f'{"="*60}')
print(f'')
print(f'UNNORMALIZED E_elec:')
print(f'  Train (Z=5-13):  min = {E_tr_min:>8.1f} Ha,  max = {E_tr_max:>8.2f} Ha')
print(f'  Test  (Z=17-27): min = {E_te_min:>8.1f} Ha,  max = {E_te_max:>8.2f} Ha')
print(f'  Test/Train range ratio: {(E_te_min - E_te_max) / (E_tr_min - E_tr_max):.1f}x')
print(f'')
print(f'NORMALIZED E_elec / Z²:')
print(f'  Train (Z=5-13):  min = {En_tr_min:>8.4f} Ha,  max = {En_tr_max:>8.4f} Ha')
print(f'  Test  (Z=17-27): min = {En_te_min:>8.4f} Ha,  max = {En_te_max:>8.4f} Ha')
print(f'  Test/Train range ratio: {(En_te_min - En_te_max) / (En_tr_min - En_tr_max):.2f}x')
print(f'')
print(f'Key point: E_elec/Z² ≈ -1/r (universal)')
print(f'  The small differences come from V_Morse/Z², which is')
print(f'  D_e/Z² ≈ {train["Z"][0]:.0f}: {(0.1+0.025*(train["Z"][0]-3))/train["Z_AB"][0]:.5f}')
print(f'          {train["Z"][-1]:.0f}: {(0.1+0.025*(train["Z"][-1]-3))/train["Z_AB"][-1]:.5f}')
print(f'          {test["Z"][0]:.0f}: {(0.1+0.025*(test["Z"][0]-3))/test["Z_AB"][0]:.5f}')
print(f'          {test["Z"][-1]:.0f}: {(0.1+0.025*(test["Z"][-1]-3))/test["Z_AB"][-1]:.5f}')
print(f'  This correction shrinks as Z grows (→ 0 for heavy atoms)')
print(f'{"="*60}')
