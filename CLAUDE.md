# CLAUDE.md — PTM Modular SAE Engine

Project memory and operational directives for Claude Code.

## Project Identity & Core Scientific Question
This thesis investigates: **Do pretrained protein language models (ESM-2 650M) emergently encode post-translational modifications (PTMs), and can Sparse Autoencoders (SAEs) disentangle them without residue collapse?**

- **Backbone**: `facebook/esm2_t33_650M_UR50D` (Target: Layer 24 activations, 1,280 dims, biological residue coordinates 1-to-1 aligned).
- **Scale Control**: `facebook/esm2_t6_8M_UR50D` (Target: Layer 4, 320 dims).
- **Domain Glossary**: Consult `CONTEXT.md` for strict terminology (`Residue Collapse`, `Chemical Stratum`, `Corpus Partition`, `Auxiliary Sequence Context`, etc.).

---

## Collaborative Division of Labor
The thesis operates on a strict two-member division of labor (see `proposal/Unified_Thesis_Plan_Modular_SAE_PTM.md`):

1. **Partner (Member 2 — Evaluation & Benchmarking)**:
   - Owns the M1–M25 statistical evaluation pipeline (`proposal/PTM_Interpretability_Implementation_Plan_V3.md`).
   - Currently running Step 3: Off-the-shelf public baseline evaluation (Arm A: InterProt TopK, Arm B: InterPLM 650M L1, Arm C: InterPLM 8M).
   - Responsible for 1,000-iteration permutation nulls, contingency tables, and proving residue collapse on public baselines.

2. **User (Member 1 — Architecture & Training)**:
   - **Owns**: Phase 1 (extraction/caching), Phase 2 (baseline ablations: TopK vs JumpReLU), Phase 3 (modular topologies: Stratified, PolySAE, CSAE), Phase 4 (Bespoke Modular SAE training: "Arm D").
   - **Current State**: Phase 0 & Phase 1 are 100% complete and verified.
   - **Remote Data Authority**: All activation shards and manifests are live at Hugging Face Dataset: `mustafa-muhaimin/ptm-sae-dataset`.

---

## Current Stage & Active Tasks (Member 1)
We are currently entering **Phase 2 & Phase 3**:

1. **Sharded DataLoader (`src/ptm_sae/data/` or `src/ptm_sae/training/`)**:
   - PyTorch `IterableDataset` / DataLoader consuming SafeTensors activation shards via `SafeTensorsReader`.
   - On-demand shard hydration from `mustafa-muhaimin/ptm-sae-dataset`.
   - Filtered strictly to `discovery_train` tokens (~5.08M tokens, 68% of corpus). `discovery_val` is for validation; `held_out` is reserved for Member 2.

2. **SAE Model Implementations (`src/ptm_sae/models/`)**:
   - `TopKSAE`: Monolithic TopK baseline ($k=32, 64$, width 4,096 / 10,240).
   - `JumpReLUSAE`: Step-threshold $\theta$ with straight-through estimator / pseudo-derivative.
   - `StratifiedModularSAE`: Modular encoder heads chemically partitioned by residue stratum (`K`, `ST`, `N`, `Y`).
   - `PolySAE`: Low-rank bilinear tensor interaction for PTM crosstalk / epistasis.
   - `BespokeModularSAE` (Arm D): Synthesis of cascaded pathway latents and polynomial crosstalk.

3. **Training Loop & Instrumentation**:
   - AdamW, LR warmup + cosine decay, gradient clipping.
   - Metrics: Reconstruction MSE, $R^2$ / Explained Variance ($\ge 85\%$), $L_0$ sparsity, dead-latent census ($<5\%$).

---

## Toolchain, Environment & Coding Standards
- **Python / Package Management**: Always use `uv run` inside the project virtual environment (`.venv`).
- **Test Suite**: Always run and verify all tests pass: `uv run pytest tests/ -v`.
- **Coding Style (per `AGENTS.md`)**:
  - High locality: avoid fracturing linear logic into tiny subfunctions.
  - Flatten nested indentation with early returns and guard clauses.
  - Table-driven logic for multi-branch classification.
  - Clean paragraph cadence; do not add ASCII art banners.
  - Zero dead or temporary code; keep commits atomic.

