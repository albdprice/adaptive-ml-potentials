# Methods

## I. Theoretical Framework

### A. Euler's Homogeneous Function Theorem and Interatomic Potentials

A scalar function $f(\mathbf{x})$ is homogeneous of degree $d$ if
$$f(s\mathbf{x}) = s^d f(\mathbf{x})$$
for all $s > 0$. By Euler's theorem, such functions satisfy
$$\mathbf{x} \cdot \nabla f = d \cdot f(\mathbf{x}).$$
For a radial pair potential $V(r)$, this reduces to $r V'(r) = d \cdot V(r)$, or in terms of force $F = -V'$,
$$r F(r) = -d \cdot V(r). \qquad (1)$$

Equation (1) holds exactly for power-law potentials: the Coulomb interaction ($V \propto r^{-1}$, $d = -1$) and London dispersion ($V \propto r^{-6}$, $d = -6$). A plot of $rF$ versus $V$ (the "virial plot") yields a straight line with slope $-d$ for any homogeneous potential.

### B. Non-Homogeneity of Realistic Potentials

Realistic covalent and van der Waals potentials are not homogeneous. The Morse potential,
$$V_{\mathrm{Morse}}(r) = D_e \bigl(1 - e^{-a(r - r_e)}\bigr)^2, \qquad (2)$$
and the Lennard-Jones (12-6) potential,
$$V_{\mathrm{LJ}}(r) = 4\varepsilon \Bigl[\bigl(\sigma/r\bigr)^{12} - \bigl(\sigma/r\bigr)^6\Bigr], \qquad (3)$$
both have position-dependent effective homogeneity degrees. Their virial plots form characteristic loops rather than straight lines, indicating that no single power law describes the interaction at all separations.

### C. Universal Binding Energy Relations

Rose, Ferrante, and Smith [Phys. Rev. B 29, 2963 (1984)] demonstrated that metallic and covalent binding energy curves collapse onto a universal form when expressed in scaled coordinates. The Rose (UBER) equation of state is
$$E(a^*) = E_c \bigl(1 - (1 + a^*)\, e^{-a^*}\bigr), \qquad (4)$$
where
$$a^* = (r - r_e) / l. \qquad (5)$$
Here $E_c$ is the cohesive energy, $r_e$ is the equilibrium distance, and $l$ is a characteristic length scale related to the curvature at equilibrium.

An analogous universality exists for the Lennard-Jones potential: defining reduced variables $r^* = r/\sigma$ and $V^* = V/\varepsilon$, all LJ curves collapse to a single master curve $V^*(r^*) = 4[(r^*)^{-12} - (r^*)^{-6}]$. The parameters $(\varepsilon, \sigma)$ fully encode the material-specific information.

### D. Adaptive Parameter Learning Hypothesis

Motivated by the success of adaptive exchange mixing in density functional approximations [Khan, Price, et al., aPBE0 (2025)], we hypothesize that machine learning models can more efficiently learn interatomic potentials by predicting the universal scaling parameters as functions of local atomic environment descriptors, rather than directly learning the energy surface $V(r)$.

The rationale is threefold:
1. **Dimensionality reduction.** Two to three scalar parameters encode the essential physics of the binding curve, compared to $\sim$50 discretized energy values for the direct approach.
2. **Bounded, smooth targets.** The scaling parameters have physical bounds and vary smoothly with atomic environment, making them easier regression targets.
3. **Physics-informed extrapolation.** The analytic functional form guarantees physically correct behavior (short-range repulsion, long-range decay) even for configurations outside the training distribution.

This is directly analogous to aPBE0, where the ML task is to learn the exchange mixing parameter $\alpha$ rather than the exchange-correlation energy directly:

| aPBE0 | Rose/UBER | Lennard-Jones |
|-------|-----------|---------------|
| $\mathbf{d} \to \alpha$ | $\mathbf{d} \to (E_c, r_e, l)$ | $\mathbf{d} \to (\varepsilon, \sigma)$ |
| $E = (1-\alpha)E_{\mathrm{PBE}} + \alpha E_{\mathrm{HF}}$ | $V = E_c(1-(1+a^*)e^{-a^*})$ | $V = 4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^6]$ |


## II. Computational Methods

### A. Model Systems

Two model potentials were studied to demonstrate the generality of the adaptive approach.

**Morse/Rose system.** Morse potentials [Eq. (2)] served as the "ground truth" binding curves, parameterized by well depth $D_e$, stiffness $a$, and equilibrium distance $r_e$. Rose parameters $(E_c, r_e^{\mathrm{Rose}}, l)$ were obtained for each Morse curve by nonlinear least-squares fitting of Eq. (4) using the Levenberg-Marquardt algorithm (SciPy `curve_fit`), with bounds $E_c \in [0.01, 20]$ eV, $r_e \in [0.5, 10]$ \AA, and $l \in [0.01, 5]$ \AA, and a maximum of 5000 function evaluations.

**Lennard-Jones system.** LJ potentials [Eq. (3)] were used as a second test case where the scaling parameters $(\varepsilon, \sigma)$ are the native parameters of the potential form, requiring no fitting step.

### B. Synthetic Dataset Generation

Descriptors $(d_1, d_2)$ served as proxies for local atomic environment features (analogous to SOAP or ACE descriptors in production ML potentials). Potential parameters were generated as linear functions of descriptors.

**Morse/Rose datasets.** Morse parameters were set as:
- $D_e = 0.5 + 2.0\, d_1$ eV
- $r_e = 1.5 + 0.5\, d_2$ \AA
- $a = 1.5$ \AA$^{-1}$ (fixed)

with descriptors sampled uniformly. Training data used $d_1, d_2 \sim \mathcal{U}(0.5, 1.5)$, yielding $D_e \in [1.5, 3.5]$ eV. Extrapolation test data used $d_1 \sim \mathcal{U}(2.5, 4.0)$, yielding $D_e \in [5.5, 8.5]$ eV. Interpolation test data used $d_1 \sim \mathcal{U}(0.7, 1.3)$, $d_2 \sim \mathcal{U}(0.6, 1.4)$. Energy curves were evaluated on 50 points spanning $r \in [0.7\, r_e,\; 3.0\, r_e]$.

**Lennard-Jones datasets.** LJ parameters were set as:
- $\sigma = 2.5 + 0.5\, d_1$ \AA
- $\varepsilon = 0.05 + 0.1\, d_2$ eV

Training data used $d_1, d_2 \sim \mathcal{U}(0.5, 1.5)$, yielding $\sigma \in [2.75, 3.25]$ \AA. Extrapolation test data used $d_1 \sim \mathcal{U}(2.5, 4.0)$, yielding $\sigma \in [3.75, 4.50]$ \AA. Interpolation test data used $d_1 \sim \mathcal{U}(0.7, 1.3)$, $d_2 \sim \mathcal{U}(0.6, 1.4)$.

Critically, all LJ energy curves were evaluated on a **fixed** radial grid of 50 points spanning $r \in [2.5, 12.0]$ \AA, independent of $\sigma$. This is essential: a $\sigma$-scaled grid (e.g., $r \in [0.9\sigma, 2.5\sigma]$) renders $\sigma/r$ constant across samples, reducing the descriptor-to-curve mapping to a trivial linear dependence on $\varepsilon$ alone. The fixed grid preserves the genuinely nonlinear dependence of $V(r)$ on $\sigma$ at fixed interatomic distances, which is the physically relevant scenario (atomic configurations in simulations have fixed coordinates, not coordinates that rescale with potential parameters).

### C. Machine Learning Models

Two approaches were compared throughout:

**Direct learning.** A Ridge regression model maps descriptors directly to the discretized potential energy curve:
$$\mathbf{d} = (d_1, d_2) \;\xrightarrow{\text{Ridge}}\; \{V(r_i)\}_{i=1}^{N_r}$$

**Adaptive learning.** A Ridge regression model maps descriptors to the universal scaling parameters, which are then transformed to energies via the analytic physics equation:
$$\mathbf{d} \;\xrightarrow{\text{Ridge}}\; \boldsymbol{\theta} \;\xrightarrow{\text{physics}}\; V_{\boldsymbol{\theta}}(r)$$
where $\boldsymbol{\theta} = (E_c, r_e, l)$ for Rose/UBER or $\boldsymbol{\theta} = (\varepsilon, \sigma)$ for LJ. Predicted parameters were clipped to physically meaningful ranges ($E_c \geq 0.01$, $r_e \geq 0.5$, $l \geq 0.01$, $\varepsilon \geq 0.001$, $\sigma \geq 0.1$) before curve reconstruction.

Ridge regression with regularization parameter $\alpha = 1.0$ was employed for both approaches. All input features and output targets were standardized (zero mean, unit variance) prior to training using `sklearn.preprocessing.StandardScaler`. The scaler was fit on training data only and applied to test data.

Ridge regression (a global linear model) was chosen deliberately over kernel ridge regression (KRR) with RBF kernels. While KRR can capture nonlinear relationships within the training domain, the RBF kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ decays to zero for test points far from training data, causing predictions to collapse to the training mean. This makes KRR fundamentally unable to extrapolate. Ridge regression, as a global linear model, can extrapolate linearly, which is sufficient to demonstrate the adaptive approach's structural advantage.

### D. Evaluation Metrics

Model performance was assessed using:

1. **Mean absolute error** (MAE):
$$\mathrm{MAE} = \frac{1}{N_{\mathrm{test}} N_r} \sum_{j=1}^{N_{\mathrm{test}}} \sum_{i=1}^{N_r} |V_j(r_i)^{\mathrm{true}} - V_j(r_i)^{\mathrm{pred}}|$$

2. **Mean squared error** (MSE):
$$\mathrm{MSE} = \frac{1}{N_{\mathrm{test}} N_r} \sum_{j=1}^{N_{\mathrm{test}}} \sum_{i=1}^{N_r} (V_j(r_i)^{\mathrm{true}} - V_j(r_i)^{\mathrm{pred}})^2$$

3. **Coefficient of determination** ($R^2$):
$$R^2 = 1 - \frac{\sum_{j,i}(V_{j,i}^{\mathrm{true}} - V_{j,i}^{\mathrm{pred}})^2}{\sum_{j,i}(V_{j,i}^{\mathrm{true}} - \bar{V}^{\mathrm{true}})^2}$$

computed over all test curves and grid points jointly.

### E. Experimental Protocols

Four systematic experiments were conducted for each model system:

**Experiment 1: Data efficiency.** Training set sizes $N \in \{10, 15, 20, 30, 40, 60, 80, 100, 120, 160\}$ were evaluated against a fixed test set of 50 samples in the extrapolation regime. For learning curve computations, 5 independent random seeds were used per training size, with results reported as mean $\pm$ standard deviation.

**Experiment 2: Interpolation vs. extrapolation.** A fixed training set ($N = 100$) was evaluated against five test regimes of increasing extrapolation severity:
- Interpolation (center): $d_1 \in [1.3, 2.2]$
- Interpolation (edges): $d_1 \in [1.0, 2.5]$
- Mild extrapolation: $d_1 \in [2.5, 3.5]$
- Moderate extrapolation: $d_1 \in [3.0, 4.0]$
- Strong extrapolation: $d_1 \in [3.5, 4.5]$

**Experiment 3: Noise robustness.** Gaussian noise with amplitude $\sigma_{\mathrm{noise}} \in \{0, 1, 2, 5, 10, 15, 20\}\%$ of the mean $|V|$ was added to training energy curves. Test data remained noise-free (extrapolation regime).

**Experiment 4: Parameter interpretability.** The quality of predicted scaling parameters was assessed individually by computing $R^2$ and MAE between true and predicted parameter values in the extrapolation regime.

### F. DFT Diatomics Validation

To validate the adaptive approach on real quantum-chemical data, binding energy curves were computed for 30 homonuclear and heteronuclear diatomic molecules spanning the first three rows of the periodic table: H$_2$, LiH, BeH, BH, CH, NH, OH, HF, NaH, MgH, AlH, SiH, PH, SH, HCl, KH, Li$_2$, N$_2$, O$_2$, F$_2$, Na$_2$, Cl$_2$, LiF, NaF, NaCl, CO, NO, CN, BF, and AlF.

**Electronic structure.** All calculations used unrestricted Kohn-Sham DFT (UKS) with the PBE exchange-correlation functional and def2-SVP basis set, as implemented in PySCF. UKS was used for all molecules (including closed-shell singlets) because restricted KS fails to describe spin-symmetry breaking at stretched geometries, causing SCF divergence for molecules such as F$_2$, Cl$_2$, and NaF. A three-level convergence strategy was employed: (1) UKS with 0.5 damping, (2) UKS without damping, and (3) UKS with Fermi smearing ($\sigma = 0.01$ Ha). The density matrix was propagated from adjacent geometry points to accelerate convergence.

**Binding curves.** For each molecule, 25 interatomic distances were sampled uniformly from $0.7\, r_{\mathrm{eq}}$ to $5.0\, r_{\mathrm{eq}}$, where $r_{\mathrm{eq}}$ is the experimental equilibrium distance. Binding energies were computed as $E_{\mathrm{bind}}(r) = E_{\mathrm{mol}}(r) - E_{\mathrm{atom}_1} - E_{\mathrm{atom}_2}$, where atomic energies were computed with UKS at the same level of theory. Rose/UBER parameters $(D_e, r_e, l)$ were obtained by nonlinear least-squares fitting of Eq. (4). Two molecules were excluded from the ML comparison due to high Rose fit residuals: H$_2$ (RMSE = 1.12 eV, UKS symmetry-breaking artifacts) and N$_2$ (RMSE = 2.09 eV, multi-reference character).

**Descriptors.** Fourteen atomic pair features were constructed for each molecule: atomic number ($Z_1$, $Z_2$), Pauling electronegativity (EN$_1$, EN$_2$), covalent radius ($r_{\mathrm{cov},1}$, $r_{\mathrm{cov},2}$), and first ionization energy (IE$_1$, IE$_2$), along with pairwise sums and differences of EN, $r_{\mathrm{cov}}$, and IE.

**Common grid.** Because different molecules have vastly different equilibrium distances (H$_2$: 0.74 \AA\ vs. Na$_2$: 3.08 \AA), a common radial grid in raw coordinates would have negligible overlap. Instead, scaled coordinates $r_s = r / r_{\mathrm{cov,sum}}$ were used, where $r_{\mathrm{cov,sum}} = r_{\mathrm{cov},1} + r_{\mathrm{cov},2}$ is the sum of covalent radii (available from the descriptors, not from adaptive parameters). This yields a scaled overlap width of 3.11, sufficient for meaningful interpolation of binding curves onto a shared grid of 50 points.

**Cross-validation protocols.** Two protocols were used:
- **Leave-one-out (LOO)**: Each of the 28 molecules is held out once; the model trains on the remaining 27 and predicts the held-out curve. This tests interpolation within the training distribution.
- **Leave-group-out (LGO)**: Molecules are split by periodic table row. Training uses molecules composed exclusively of row 1--2 atoms (H through Ne; 15 molecules). Testing uses molecules containing at least one row 3+ atom (Na through K; 13 molecules). This tests extrapolation to heavier elements not seen during training.

Both approaches used Ridge regression with $\alpha = 1.0$ and StandardScaler, identical to the synthetic experiments.

### G. Software and Reproducibility

All calculations were performed using Python 3.11 with NumPy, SciPy 1.12 (nonlinear fitting), scikit-learn 1.4 (Ridge regression, StandardScaler), and Matplotlib 3.8 (visualization). DFT calculations used PySCF 2.5. Random seeds were fixed for reproducibility. Learning curve computations and DFT scans were performed on the Compute Canada Cedar cluster. All source code and data files are available at [repository URL].
