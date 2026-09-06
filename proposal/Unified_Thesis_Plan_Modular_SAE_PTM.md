# Unified Thesis Specification: Modular Sparse Autoencoders for Protein PTM Disentanglement and Mechanistic Interpretability

## Executive Summary & Collaborative Architecture
This document unifies the 2-member undergraduate thesis into a coherent, publication-grade master plan:
* **Member 1 (Architectural Innovation & Training)**: Responsible for Phase 1 (activation extraction/caching from ESM-2 650M Layer 24), Phase 2 (baseline ablations: Monolithic TopK vs. JumpReLU), Phase 3 (modular topology exploration: Hierarchical Modular, Cascaded CSAE, PolySAE bilinear tensor interactions, and Matryoshka), and Phase 4 (synthesis of the novel **Cascaded-Polynomial Modular SAE [Bespoke Modular SAE]**).
* **Member 2 (Statistical Evaluation, Causality & Benchmarking)**: Responsible for the complete 25-module statistical evaluation pipeline (`PTM_Interpretability_Implementation_Plan_V3.md`): data harmonization, leak-free 50% sequence-identity CD-HIT partitioning, 2x2 contingency tables, 1,000-iteration chemistry-stratified permutation nulls, Bonferroni FWER control, and causal activation-patching gates.
* **The Inter-Member Convergence Contract**: Member 2 immediately benchmarks 21 public off-the-shelf checkpoints (InterProt TopK, InterPLM L1 650M, InterPLM 8M) to establish the empirical baseline and quantify failure modes. Concurrently, Member 1 executes the ablation suite and trains the novel Bespoke Modular SAE. In Months 6–9, Member 1's custom modular checkpoints are ingested into Member 2's pipeline as "Arm D", proving whether the modular architecture eliminates residue collapse and resolves non-linear PTM crosstalk.

---

## Part 1: Member 1 Workstream — Modular SAE Architectures & Ablation Suite

### 1.1 The Theoretical Bottleneck: Monolithic Superposition & Residue Collapse
Standard sparse autoencoders decompose dense activations $x \in \mathbb{R}^d$ via linear dictionary reconstruction:
$$\hat{x} = W_d h + b_d, \quad h = f(W_e(x - b_d) + b_e)$$
In protein language models (e.g., ESM-2 650M), this linear additive assumption induces two catastrophic failure modes:
1. **Residue-Identity Collapse**: Features fire generically for amino acid identity (e.g., any lysine $	ext{K}$) rather than functional modification states (e.g., ubiquitinated lysine in a degron).
2. **Failure on Combinatorial Epistasis (PTM Crosstalk)**: In cellular signaling, multiple modifications act as logical AND/OR gates (e.g., phosphorylation at Ser10 priming acetylation at Lys14 on Histone H3). Flat monolithic SAEs cannot represent pairwise or higher-order dependencies without dedicating distinct latents to compound states.

### 1.2 Multi-Stage Ablation Progression
Member 1 systematically executes four progressive experimental phases:

#### Phase 1: Activation Extraction & Caching
* **Backbone**: Frozen `esm2_t33_650M_UR50D` (1,280-dim activations).
* **Layer**: Layer 24 (the functional/modification semantic peak).
* **Corpus**: ~20,400 reviewed human Swiss-Prot sequences ($\le 1,022$ residues). Activations are pre-extracted to local storage in `fp16` format (~45 GB), decoupling SAE training from PLM inference.

#### Phase 2: Baseline Monolithic Ablation & Public Checkpoint Benchmarking
* **Pretrained Public Baselines**:
  * `liambai/InterProt-ESM2-SAEs` (Arm A): TopK architecture ($k=64$, 4,096 dictionary = 3.2x expansion), Layer 24 and 33 of `esm2_t33_650M_UR50D`.
  * `Elana/InterPLM-esm2-650m` (Arm B): L1-ReLU architecture (10,240 dictionary = 8x expansion), Layer 24 and 33 of `esm2_t33_650M_UR50D`.
  * `Elana/InterPLM-esm2-8m` (Arm C): L1-ReLU scale control (10,240 dictionary = 32x expansion), Layer 4 and 6 of `esm2_t6_8M_UR50D`.
* **Newly Trained Ablations**: Monolithic JumpReLU SAE on cached Layer 24 activations to benchmark step-thresholding against InterPLM's continuous L1 shrinkage.
* **Objective**: Establish the empirical performance floor across reconstruction fidelity ($R^2 > 0.92$), L0 sparsity, dead-feature census (<5%), and residue-dominance collapse rate across monolithic architectures.

#### Phase 3: Modular Topologies Benchmark
1. **Hierarchical Modular Modular SAEs**: Decomposing representations into sparsely communicating domain modules dedicated to distinct enzyme families (kinases, ubiquitin ligases, acetyltransferases).
2. **Cascaded Sparse Autoencoders (CSAE)**: Training a second-tier SAE on the decoder weights of the first-tier SAE to learn "concepts of concepts" (hierarchical pathway-level abstractions).
3. **Hierarchical & Multi-Scale Grouping (HiSAE & Matryoshka SAEs)**:
   * **HiSAE**: Enforces conditional parent-child gating where fine-grained modification latents fire only when higher-level parent structural/sequence modules activate.
   * **Matryoshka SAEs**: Learns nested representations of expanding capacity, constraining early modular dimensions to broad structural folds and later dimensions to granular PTM context.
4. **Compositional Logic via PolySAE**: Incorporating low-rank bilinear tensor factorization ($\hat{x} = W_d h + \sum_{i,j} \mathcal{T}_{i,j} h_i h_j$) to capture pairwise epistatic interactions without statistical co-occurrence noise.
4. **Hierarchical & Multi-Scale Grouping (HiSAE & Matryoshka)**: Nested dictionary scales enforcing parent-child activation dependencies from residue motifs to structural domains.

#### Phase 4: Formulation of the Novel Bespoke Modular SAE Architecture
Synthesizing the ablation findings into a **Cascaded-Polynomial Modular Sparse Autoencoder (Bespoke Modular SAE)**:
* Chemically-partitioned modular encoder heads for modifiable residue strata ($	ext{K, S/T, N, C, Y}$).
* Cascaded pathway-abstraction latent layer.
* Low-rank bilinear interaction tensor modeling combinatorial PTM crosstalk across modules.

---

## Part 2: Member 2 Workstream — The Statistical Interpretability Engine

Member 2 executes the 25-module execution specification from `PTM_Interpretability_Implementation_Plan_V3.md`:

### 2.1 Data Invariants & Leakage Control
* **Data Sources**: Human Swiss-Prot, dbPTM 2025, CPLM 4.0, qPTM, O-GlcNAcAtlas 4.0, N-GlyDE. Strictly experimental evidence tiers; homology transfers and text-mining dropped.
* **Split Scheme**: 50% sequence identity clustering via CD-HIT into Discovery (80%) and Held-Out (20%) partitions, stratified across PTM classes.

### 2.2 Statistical Validation & Permutation Nulls
* **2x2 Contingency Matrix**: Built from exact activation suffix sums across a sweep of activation thresholds $	au \in [0.0, 0.80]$.
* **Metrics**: Fold Enrichment, Selective Odds Ratio, Cliff's Delta, Area Under Precision-Recall (AUPRC).
* **Chemical Stratum-Preserving Permutation Null ($N=1,000$)**: Shuffling positive labels strictly within the corresponding amino-acid background stratum (e.g., shuffling phosphorylation labels only across Ser/Thr residues) to guarantee latents detect modification biochemistry rather than amino acid prevalence.
* **Family-Wise Error Control**: Strict Bonferroni correction per dictionary head at $lpha = 0.05$.

### 2.3 Quality Gates & Causal Verification
* **Fidelity Gate**: Explained variance $\ge 85\%$, dead latents $< 5\%$.
* **Residue-Dominance Gate**: Latents where $\ge 7$ of the top-10 activating positions share the same amino acid are excluded as unselective sequence detectors.
* **Causal Activation Patching**: In silico steering and ablation demonstrating that clamping modular latents causally modulates downstream PTM functional predictions.

---

## Part 3: Collaborative Timeline (10 Months)

```
Month 01-02: [Member 1] Pre-extract ESM-2 Layer 24 activations to disk.
             [Member 2] Harmonize proteomic databases, run CD-HIT 50% clustering, build label matrices.
Month 03-04: [Member 1] Implement & train Phase 2 baseline ablations (TopK vs. JumpReLU).
             [Member 2] Run M1-M15 statistical evaluation on public baselines (InterProt, InterPLM 650M/8M).
Month 05-06: [Member 1] Implement Phase 3 modular candidates (Hierarchical, CSAE, PolySAE, Matryoshka).
             [Member 2] Run 1,000-iteration permutation nulls on baselines; identify residue-collapse rates.
Month 07-08: [Member 1] Formulate, implement, and train the novel Bespoke Modular SAE architecture.
             [Member 2] Ingest Member 1's Bespoke Modular SAE into M1-M25 pipeline as "Arm D".
Month 09-10: [Joint] Complete causal activation-patching experiments, compile Project Completion Report (PCR),
             prepare RISE poster, and draft conference submission manuscript (IEEE BIBM / ACM BCB).
```
