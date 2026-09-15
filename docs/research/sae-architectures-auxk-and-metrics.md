# Technical Reference: SAE Architectures, AuxK, Geometric Median, and Telemetry

This document provides a rigorous mathematical and operational reference for the Phase 2 baseline Sparse Autoencoders (SAEs), their initialization priors, dead-latent mitigation mechanics (AuxK), and training-time validation telemetry.

---

## 1. Mathematical Anatomy of the Four Baseline SAE Architectures

All SAEs in this repository operate on activations $x \in \mathbb{R}^{d_{\text{in}}}$ (Layer 24 of `esm2_t33_650M_UR50D`, $d_{\text{in}} = 1,280$) and map them to an overcomplete latent space $z \in \mathbb{R}^{d_{\text{hidden}}}$ ($d_{\text{hidden}} = 4,096$ or $10,240$), reconstructing $\hat{x} \in \mathbb{R}^{d_{\text{in}}}$:

$$\hat{x} = z W_{\text{dec}} + b_{\text{dec}}$$

where $W_{\text{dec}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{in}}}$ has unit-norm rows ($\|W_{\text{dec}, i}\|_2 = 1$), and $b_{\text{dec}} \in \mathbb{R}^{d_{\text{in}}}$ is the decoder bias.

```
                   x (Activation, d_in = 1280)
                              │
                    ┌─────────┴─────────┐
                    │  Centering:       │
                    │  x̃ = x - b_dec    │
                    └─────────┬─────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
     [ TopK ]            [ JumpReLU ]         [ Gated SAE ]
  Exact top-k         Learned step-cutoff   Dual-path encoder
  ReLU(x̃ W_e + b_e)    z̃ · 1[z̃ > θ]          Gate(x̃) ⊙ Mag(x̃)
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    Latents z (d_hidden)
                              │
                    ┌─────────┴─────────┐
                    │     Decoder       │
                    │ x̂ = z W_dec + b_d │
                    └───────────────────┘
```

---

### Architecture 1: Monolithic TopK SAE (Gao et al., 2024)

* **Primary Source**: Gao, L., et al. (OpenAI, 2024). *Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093.
* **Core Problem It Solves**: Eliminates shrinkage bias. In classical $L_1$-penalized SAEs, the $L_1$ penalty constantly exerts downward pressure on firing latents, shrinking large activations away from their true magnitudes and forcing an artificial trade-off between sparsity and reconstruction.
* **Mathematical Formulation**:
  1. Pre-activation:
     $$\tilde{z} = \text{ReLU}((x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}})$$
  2. Latent code selection:
     $$z_i = \begin{cases} \tilde{z}_i & \text{if } \tilde{z}_i \in \text{TopK}(\tilde{z}, k) \\ 0 & \text{otherwise} \end{cases}$$
  3. Objective function:
     $$\mathcal{L}_{\text{TopK}} = \|x - \hat{x}\|_2^2$$
* **Key Properties**:
  - $L_0$ is constant and strictly equals $k$ for every token.
  - No sparsity hyperparameter tuning or $L_1$ schedule is needed.
  - **Limitation**: Highly uneven complexity across protein sequences is forced into the exact same $k$ budget. If a latent never makes the top-$k$, its gradient is identically zero, making it susceptible to permanent latent death without AuxK.

---

### Architecture 2: JumpReLU SAE (Rajamanoharan et al., 2024)

* **Primary Source**: Rajamanoharan, S., et al. (Google DeepMind, 2024). *Jumping Ahead: Improving Reconstruction and Sparsity with JumpReLU Sparse Autoencoders*. arXiv:2407.14435.
* **Core Problem It Solves**: Achieves variable per-token sparsity without $L_1$ shrinkage bias.
* **Mathematical Formulation**:
  1. Activation function with learned step-threshold $\theta_i = \exp(\log \theta_i) > 0$:
     $$z_i = \tilde{z}_i \cdot \mathbf{1}[\tilde{z}_i > \theta_i], \quad \text{where } \tilde{z} = (x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}}$$
  2. Training objective with direct pseudo-$L_0$ penalty:
     $$\mathcal{L}_{\text{JumpReLU}} = \|x - \hat{x}\|_2^2 + \lambda_{L_0} \sum_{i=1}^{d_{\text{hidden}}} \mathbf{1}[\tilde{z}_i > \theta_i]$$
* **The Straight-Through Estimator (STE) Pseudo-Derivative**:
  The Heaviside step $\mathbf{1}[\tilde{z} > \theta]$ is non-differentiable (zero derivative everywhere, undefined at $\theta$). JumpReLU replaces the true Dirac delta derivative with a rectangular kernel of bandwidth $\epsilon$:
  $$\frac{\partial z_i}{\partial \theta_i} \approx -\frac{\theta_i}{\epsilon} \mathbf{1}\left[|\tilde{z}_i - \theta_i| < \frac{\epsilon}{2}\right]$$
  Since thresholds are parameterized in log-space ($\phi_i = \log \theta_i$) to guarantee $\theta_i > 0$, the chain rule applies:
  $$\frac{\partial \mathcal{L}}{\partial \phi_i} = \frac{\partial \mathcal{L}}{\partial \theta_i} \cdot \theta_i = -\frac{\theta_i^2}{\epsilon} \mathbf{1}\left[|\tilde{z}_i - \theta_i| < \frac{\epsilon}{2}\right] \left( \nabla_{z_i} \mathcal{L} + \frac{\lambda_{L_0}}{\theta_i} \right)$$
* **Key Properties**: Once $\tilde{z}_i > \theta_i$, the activation passes unattenuated (zero shrinkage).

---

### Architecture 3: BatchTopK SAE (Bussmann, 2024)

* **Primary Source**: Bussmann, B. (2024). *BatchTopK: A Simple Modification for Sparsity-Constrained Autoencoders*.
* **Core Problem It Solves**: Protein sequences exhibit extreme biological heterogeneity. Rigid per-token $k$ forces uninformative amino acids (e.g. poly-alanine tracts, disordered flexible linkers) to recruit $k$ latents while starving information-dense catalytic pockets or post-translationally modified sites.
* **Mathematical Formulation**:
  1. For a mini-batch of $B$ tokens, flatten all pre-activations into a single vector of length $B \cdot d_{\text{hidden}}$.
  2. Select the top $(B \cdot k)$ activations jointly across the whole batch:
     $$z = \text{TopK}_{B \cdot k}\left(\text{ReLU}((X - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}})\right)$$
  3. Tokens with high semantic complexity recruit $> k$ latents; low-complexity tokens recruit $< k$ latents, while the batch average remains exactly $k$.
* **Inference / Evaluation Mechanics**:
  During single-protein inference or validation, there is no large batch to pool over. BatchTopK tracks an exponential moving average (EMA) running threshold during training:
  $$\theta_{\text{EMA}}^{(t)} = \beta \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \mathbb{E}_{\text{tokens}}\left[ \min_{i: z_i > 0} z_i \right]$$
  At eval time, it operates like a JumpReLU model using $\theta_{\text{EMA}}$.

---

### Architecture 4: Gated SAE (Rajamanoharan et al., 2024a)

* **Primary Source**: Rajamanoharan, S., et al. (Google DeepMind, 2024a). *Improving Dictionary Learning with Gated Sparse Autoencoders*. arXiv:2404.16014.
* **Core Problem It Solves**: In standard SAEs, encoder weights perform two conflicting jobs: determining *which* features fire (detection) and *how strongly* they fire (magnitude). Gated SAE decouples them without requiring straight-through estimators.
* **Mathematical Formulation**:
  1. **Gating Path** (detects feature presence):
     $$\pi_{\text{gate}} = (x - b_{\text{dec}}) W_{\text{gate}} + b_{\text{gate}}, \quad g = \mathbf{1}[\pi_{\text{gate}} > 0]$$
  2. **Magnitude Path** (estimates intensity, weight-tied via per-latent parameter $r_{\text{mag}}$):
     $$W_{\text{mag}} = W_{\text{gate}} \odot \exp(r_{\text{mag}}), \quad m = \text{ReLU}((x - b_{\text{dec}}) W_{\text{mag}} + b_{\text{mag}})$$
  3. **Latent Code**:
     $$z = g \odot m$$
* **Dual-Objective Training without STE**:
  Instead of differentiating through the hard step $\mathbf{1}[\pi_{\text{gate}} > 0]$, Gated SAE trains the gate using an auxiliary reconstruction with a frozen decoder:
  $$\hat{x}_{\text{aux}} = \text{ReLU}(\pi_{\text{gate}}) W_{\text{dec}}^{\text{detached}} + b_{\text{dec}}^{\text{detached}}$$
  $$\mathcal{L}_{\text{Gated}} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{Main MSE}} + \lambda_{\text{aux}} \underbrace{\|x - \hat{x}_{\text{aux}}\|_2^2}_{\text{Auxiliary Gate Loss}} + \lambda_{L_1} \underbrace{\sum_{i=1}^{d_{\text{hidden}}} \text{ReLU}(\pi_{\text{gate}, i})}_{\text{Gate Sparsity}}$$
* **Why it Works**: The frozen decoder prevents the auxiliary task from corrupting the dictionary directions, while providing rich reconstruction gradients directly to $W_{\text{gate}}$ and $b_{\text{gate}}$.

---

## 2. AuxK: Dead Latent Revival Mechanism

* **Primary Source**: Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders*. Section 2.2.
* **The Dead Latent Crisis in Hard Selection**:
  In TopK and BatchTopK, latents outside the top-$k$ have an exact zero activation: $z_j = 0$. Consequently:
  $$\frac{\partial \mathcal{L}}{\partial W_{\text{enc}, j}} = \frac{\partial \mathcal{L}}{\partial z_j} \frac{\partial z_j}{\partial W_{\text{enc}, j}} = 0 \cdot \frac{\partial z_j}{\partial W_{\text{enc}, j}} = 0$$
  If a latent falls out of the top-$k$ across all tokens in a training window, it never receives gradient again. It is permanently dead.
* **The AuxK Formulation**:
  AuxK revives dead latents by assigning them an auxiliary task: **reconstruct the residual error that the main model failed to capture**.
  1. Compute the detached residual of the primary forward pass:
     $$e = (x - \hat{x}).\text{detach}()$$
  2. Identify the dead latents using a token-windowed census ($\text{tokens\_since\_fired} > T_{\text{window}}$, e.g. $10^7$ tokens). Let $\mathcal{D}$ be the set of dead latent indices.
  3. Select the top-$k_{\text{aux}}$ activations (heuristic: $k_{\text{aux}} = \text{nearest\_power\_of\_two}(d_{\text{in}} / 2) = 512$) strictly from the dead set $\mathcal{D}$:
     $$z_{\text{aux}} = \text{TopK}_{k_{\text{aux}}}\left( \text{ReLU}(\tilde{z}) \odot \mathbf{1}_{i \in \mathcal{D}} \right)$$
  4. Predict the residual:
     $$\hat{e} = z_{\text{aux}} W_{\text{dec}} \quad (\text{note: no } b_{\text{dec}} \text{ because } e \text{ is already centered})$$
  5. Auxiliary Loss:
     $$\mathcal{L}_{\text{auxk}} = \|e - \hat{e}\|_2^2$$
* **Why Detaching $e$ is Essential**:
  Because $e$ is detached, the auxiliary loss does not backpropagate through the main model's predictions. The alive latents continue optimizing primary reconstruction without interference, while dead latents are actively pulled toward the largest unexplained directions of variance. Once a revived latent begins firing in the main path, it naturally exits $\mathcal{D}$ and stops receiving AuxK gradients.

---

## 3. Geometric Median Prior for Decoder Bias

* **Primary Sources**:
  - Anthropic (Bricken et al., 2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*.
  - Weiszfeld, E. (1937). *Sur le point pour lequel la somme des distances de $n$ points donnés est minimum*. Tohoku Math. J.
* **The Representation Dilemma**:
  Language model activations do not sit centered at the origin; they occupy an eccentric hyper-ellipsoid with a large non-zero mean offset. If $b_{\text{dec}} = 0$, dictionary latents waste representational capacity simply reconstructing this global offset rather than functional variations.
* **Why the Geometric Median Over the Arithmetic Mean?**
  The arithmetic mean $\mu = \frac{1}{N} \sum_{i=1}^N x_i$ minimizes squared Euclidean distance:
  $$\mu = \arg\min_m \sum_{i=1}^N \|x_i - m\|_2^2$$
  Because distances are squared, the arithmetic mean is heavily distorted by rare outlier activations (e.g. rare tokens, special tokens `<cls>`, `<eos>`, or extreme hydrophobic clusters).
  In contrast, the **Geometric Median** minimizes the sum of unsquared Euclidean distances:
  $$m^* = \arg\min_m \sum_{i=1}^N \|x_i - m\|_2$$
  It has a breakdown point of $50\%$, meaning up to half the data can be arbitrary outliers without pulling $m^*$ to infinity.
* **Weiszfeld's Algorithm**:
  Since no closed-form solution exists in higher dimensions, Weiszfeld's algorithm solves for $m^*$ via Iteratively Reweighted Least Squares (IRLS):
  $$m^{(t+1)} = \frac{\sum_{i=1}^N \frac{x_i}{\|x_i - m^{(t)}\|_2 + \epsilon}}{\sum_{i=1}^N \frac{1}{\|x_i - m^{(t)}\|_2 + \epsilon}}$$
  Points far from the current estimate receive small weights ($1/\text{distance}$), causing the estimate to rapidly converge to the true centroid of the dense activation manifold.
* **Role in Centering**:
  By setting $b_{\text{dec}} = m^*$ prior to training, the encoder input $(x - b_{\text{dec}})$ centers the data cloud. Dictionary latents can then strictly learn sparse directional deviations corresponding to biological motifs and PTM states.

---

## 4. Telemetry & W&B Metrics Dictionary

During training, metrics are streamed to Weights & Biases and local notebooks across three distinct categories:

### A. Optimization & Reconstruction Fidelity
| Metric | Code Key | Target / Healthy Range | Interpretation |
| :--- | :--- | :--- | :--- |
| **Validation MSE** | `val/mse` | Monotonically decreasing | Mean squared reconstruction error on held-out `discovery_val` proteins. |
| **Explained Variance ($R^2$)** | `val/explained_variance` | $\ge 85\%$ ($0.85 - 0.95$) | Fraction of activation variance explained: $1 - \frac{\text{Var}(x - \hat{x})}{\text{Var}(x)}$. Core fidelity gate of the thesis. |
| **Cosine Similarity** | `val/cosine_sim_mean`, `p10` | Mean $\ge 0.92$, $p10 \ge 0.85$ | Angular alignment between $x$ and $\hat{x}$. $p10$ checks the worst $10\%$ of tokens to ensure no biological subspaces are lost. |
| **Decoder Pre-Norm Mean** | `train/decoder_pre_norm_mean` | $\approx 1.00 \pm 0.05$ | Mean norm of $W_{\text{dec}}$ before post-step unit-norm projection. If it drops $<0.9$ or exceeds $>1.2$, learning rates are fighting constraints. |

### B. Sparsity & Dictionary Health
| Metric | Code Key | Target / Healthy Range | Interpretation |
| :--- | :--- | :--- | :--- |
| **Active Sparsity ($L_0$)** | `train/l0`, `val/mean_l0` | Target $k \pm 5\%$ | Mean number of active latents per residue. Exactly $k$ for TopK; emergent for JumpReLU and Gated. |
| **Dead Latent Fraction** | `val/dead_latent_fraction` | $< 5.0\%$ | Fraction of latents that have not fired for $10^7$ tokens. Spike indicates latent death; verifies AuxK efficacy. |
| **Alive Latent Jaccard** | `val/alive_latent_jaccard` | $\to 95\% - 100\%$ | Jaccard similarity $J(S_t, S_{t-1}) = \frac{|S_t \cap S_{t-1}|}{|S_t \cup S_{t-1}|}$ of active latent sets between evals. Measures feature basis convergence. |
| **Feature Density Histogram** | `val/feature_density_histogram` | Log-normal spread | Distribution of latent firing frequencies across $\log_{10}$ bins. Confirms presence of specific low-frequency functional features. |

### C. The Biological Interpretability Canaries (Phase 2 Diagnostic Suite)
| Metric | Code Key | Healthy Range | Diagnostic Purpose |
| :--- | :--- | :--- | :--- |
| **Residue Dominance Collapse Rate** | `residue_dominance/collapse_rate` | $< 25\%$ | **PTM-label-free test**: checks if $\ge 70\%$ of a latent's top-10 activating residues share a single amino acid. High values indicate the model collapsed into generic amino acid detectors. |
| **Stratum Concentration Ratio** | `collapse_check/by_stratum/*/mean_concentration_ratio` | $\ge 2.0\times$ baseline | Measures whether latents fire on modified residues at elevated rates compared to the chemical stratum's background base rate. |
| **PTM-Specific Selectivity** | `collapse_check/by_ptm_type/*/best_latent_ratio` | High ratio, small `n_latents_near_best` | Tests if a latent specifically fires for one PTM type (e.g. Phosphoserine) vs broadly firing for any modification on that amino acid. |
| **Shortcut Learning Suspects** | `collapse_check/by_negative_tier/*/n_latents_shortcut_suspect` | Close to 0 | Flags latents firing on curated "hard negatives" (unmodified residues chemically similar to modified ones) at $>80\%$ rate. |

