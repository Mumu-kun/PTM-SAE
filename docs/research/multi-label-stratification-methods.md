# Comparative Research: Multi-Label Stratification Methods & Why Greedy Target Balancing Dominates

**Target Question**: *Why do we use greedy target balancing for multi-label data splitting, and what are the alternative methods?*

**Scope**: Homology-reducing sequence cluster partitioning, multi-label PTM distribution preservation, and dataset splitting algorithms in machine learning.

---

## 1. Executive Summary & Comparative Matrix

When partitioning protein clusters into `discovery` (e.g., 80%) and `held_out` (e.g., 20%), two competing constraints exist:
1. **Strict Homology Grouping**: All proteins in a 50% sequence identity cluster must stay together to prevent representation leakage.
2. **Multi-Label Balance**: Each cluster contains proteins modified by multiple overlapping PTMs (e.g. 15 Phosphorylations, 3 Ubiquitinations, 1 Acetylation). The ratio of *every* PTM type—especially rare ones like O-GlcNAcylation—must be preserved across both partitions.

Across literature and production ML libraries, **five primary splitting paradigms** exist:

| Method | Algorithm Class | Optimality Guarantee | Computational Complexity | Handles Rare Labels? | Handles Cluster / Group Constraints? | Primary Sources / Implementations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Greedy Iterative Stratification** | Deterministic Greedy Heuristic | Near-optimal heuristic | **$O(N \cdot L)$ (Milliseconds)** | **Excellent** (prioritizes rarest labels first) | Native (run on cluster aggregate vectors) | Sechidis et al. (ECML-PKDD 2011); Szymański & Kajdanowicz (2017); `iterative-stratification` |
| **2. Mixed-Integer Linear Programming (MILP)** | Exact Mathematical Optimization | **Globally Optimal** | $O(2^N)$ worst-case (Seconds to minutes via Branch-and-Cut) | **Guaranteed** within solver tolerance | Native (enforces integer cluster assignments) | SciPy `milp` (HiGHS solver), PuLP, Gurobi |
| **3. Simulated Annealing / Genetic Algorithms** | Stochastic Metaheuristics | Asymptotically optimal | $O(K \cdot N \cdot L)$ (Iterative swaps, seconds to minutes) | Good (if included in fitness loss) | Native (swaps entire cluster IDs) | Global optimization literature; customized research pipelines |
| **4. Label Powerset Stratification** | Multi-class Reduction | Sub-optimal | $O(N \log N)$ | **Catastrophic Failure** (combinatorial explosion, $2^L$ singletons) | Poor | Standard `scikit-learn` `StratifiedKFold` on powerset strings |
| **5. Dominant-Label Stratification** | Heuristic Approximation | Single-label only | $O(N)$ | **Catastrophic Failure** (starves rare secondary labels) | Native | Naive `sklearn.model_selection.GroupKFold` |

---

## 2. Why Greedy Target Balancing (Iterative Stratification)?

### A. The Core Invariant: Prioritizing the Rarest Label First
The algorithm was formalized by **Sechidis, Tsoumakas, & Vlahavas (2011)** in *On the Stratification of Multi-label Data* (ECML-PKDD 2011) and extended to grouped/higher-order distributions by **Szymański & Kajdanowicz (2017)**:

1. **The Scarcity Bottleneck**:
   In our PTM dataset, Phosphorylation has ~250,000 sites, while O-GlcNAcylation or Methylation might have only ~800 sites.
   If you assign clusters containing Phosphorylation first, partitions will quickly fill up to their 80% and 20% capacity quotas before you ever look at O-GlcNAcylation. By the time you reach O-GlcNAcylation, `discovery` might be full, forcing all O-GlcNAc sites into `held_out` (or vice versa).
2. **The Greedy Resolution**:
   Iterative stratification solves this by dynamically sorting labels by **scarcity**:
   - At each step, it finds the label $c^*$ with the **fewest remaining unassigned positive instances**.
   - It finds the cluster possessing $c^*$ that has the fewest *other* active labels (most specialized cluster).
   - It assigns that cluster to the partition currently suffering the **largest percentage deficit** for label $c^*$ relative to its target quota.
   - It updates all label counts and repeats until all clusters are assigned.

### B. Grouped / Clustered Extension (Homology Clusters)
Because sequence clustering groups homologous proteins together (e.g. 50% identity via MMseqs2), the assignment unit is the **Cluster**, not the protein:
- Input instance $i$ is a `cluster_id`.
- Feature label vector $y_i$ is the **sum of PTM sites** across all proteins in cluster $i$:
  $$y_{i, c} = \sum_{p \in \text{Cluster}_i} \text{Count}_{p}(c)$$
- Running Iterative Stratification directly on the cluster label profiles guarantees:
  1. **Zero Homology Leakage**: No cluster is ever split across partitions.
  2. **Multi-Label Balance**: Both `discovery` and `held_out` receive their proportional share of all PTM types.

---

## 3. Detailed Comparison with Alternative Methods

### Alternative 1: Mixed-Integer Linear Programming (MILP)
* **How It Works**:
  Defines binary decision variables $x_{i} \in \{0, 1\}$ indicating whether cluster $i$ is placed in `discovery` ($x_i = 1$) or `held_out` ($x_i = 0$).
  Formulates an objective minimizing the weighted absolute deviation from target counts across all PTM types:
  $$\min \sum_{c=1}^L w_c \left| \sum_{i=1}^{N_{\text{clusters}}} x_i y_{i, c} - p_{\text{target}} Y_c \right| + w_{\text{size}} \left| \sum_{i=1}^{N_{\text{clusters}}} x_i L_i - p_{\text{target}} L_{\text{total}} \right|$$
* **Pros**: Provably finds the globally optimal mathematical split. It can simultaneously balance PTM site counts AND total residue counts ($L_i$).
* **Cons**: Formulating piecewise absolute value objectives in MILP requires slack variables ($2 \times L$ auxiliary variables). For 12,000+ clusters, open-source solvers (HiGHS / CBC) can take minutes to converge compared to milliseconds for greedy heuristics.

### Alternative 2: Stochastic Optimization (Simulated Annealing)
* **How It Works**:
  Initializes partitions randomly or via greedy assignment. Computes a global penalty function (e.g. sum of Kullback-Leibler divergences or mean squared percentage errors across label proportions). Iteratively proposes swapping a random cluster from `discovery` to `held_out`. Swaps that reduce discrepancy are accepted; swaps that increase discrepancy are accepted with probability $e^{-\Delta E / T}$.
* **Pros**: Highly customizable. Allows adding complex biological constraints (e.g., ensuring both partitions have equal distributions of structural classes or enzyme families).
* **Cons**: Non-deterministic (stochastic; outputs vary by random seed) and computationally slower (often requiring 10,000–50,000 iterations).

### Alternative 3: Label Powerset (LP) Stratification (The Naive Baseline)
* **How It Works**:
  Converts the multi-label set into a single composite string (e.g., `{Phosphorylation, Ubiquitination}` $\to$ `"Phos+Ub"`). Runs standard single-class `StratifiedKFold`.
* **Why It Fails for PTMs**:
  With 10+ PTM types, there are $2^{10} = 1,024$ possible combinations. High-modification proteins (e.g., P53, Histone H3) possess unique combinatorial cocktails that appear only **once** in the entire proteome (singletons). A single cluster with a unique combination cannot be split across both partitions, causing the algorithm to crash or dump all rare combinations into one bucket.

### Alternative 4: Dominant-Label Splitting (The Single-Label Fallback)
* **How It Works**:
  Each cluster is assigned a single label based on its most frequent modification (e.g. if a cluster has 10 Phospho sites and 1 Acetyl site, it is classified solely as `"Phosphorylation"`). Standard grouped stratification is applied.
* **Why It Fails for PTMs**:
  Phosphorylation accounts for $>70\%$ of all known PTM sites. Under dominant-label splitting, almost every multi-modified cluster is labeled `"Phosphorylation"`. Secondary PTMs (acetylation, methylation, O-GlcNAcylation) become invisible "passengers" that get distributed entirely at random, leading to extreme imbalance in the held-out test set.

---

## 4. Conclusion & Recommendation for PTM Engine

For our thesis architecture:
1. **Primary Algorithm**: **Greedy Iterative Stratification** (Sechidis / Szymański) applied to **Homology Clusters**.
   - It is fast, deterministic, handles severe class imbalance (from 250k Phosphorylations to 800 O-GlcNAcylations), and directly satisfies **CF-16**.
2. **Audit Verification**:
   After the greedy split finishes, emit a **Stratification Balance Diagnostic Table**:
   - Compare the realized proportion in `discovery` vs `held_out` for each individual PTM type.
   - If any rare PTM deviates by more than a pre-stated tolerance (e.g., target is 20.0%, realized is $<16\%$ or $>24\%$), trigger an automated fine-tuning pass using **SciPy MILP** or a local swap heuristic to rebalance the boundary clusters.
