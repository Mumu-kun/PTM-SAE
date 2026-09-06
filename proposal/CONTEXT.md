# Domain Model: PTM Mechanistic Interpretability & Modular SAEs

## Core Terminology

### Protein Language Model (PLM)
Pretrained transformer models operating on raw amino acid sequences (e.g., ESM-2 family `esm2_t33_650M_UR50D`, `esm2_t6_8M_UR50D`). Hidden representations at intermediate layers encode biophysical, structural, and evolutionary constraints.

### Sparse Autoencoder (SAE)
An unsupervised neural network trained to reconstruct model activations $x \in \mathbb{R}^d$ through an overcomplete hidden dictionary $f \in \mathbb{R}^m$ ($m \gg d$) under an explicit sparsity constraint ($L_0 \ll m$). Decomposes polysemantic activations into monosemantic, interpretable latent directions.

### Post-Translational Modification (PTM)
Covalent enzymatic modifications of amino acid side chains (e.g., phosphorylation on Ser/Thr/Tyr, ubiquitination/acetylation on Lys, N-glycosylation on Asn). Function as cellular switches regulating signaling, localization, and degradation.

### Residue Collapse (Failure Mode)
A failure mode in monolithic protein SAEs where latents fire selectively for amino acid identity (e.g., 'any lysine') rather than specific biological context or modification state (e.g., 'ubiquitinated lysine in a degron motif').

### Modular SAE
An SAE architecture featuring decoupled, specialized sub-dictionaries (either chemically partitioned by modifiable residue class or dynamically routed via MoE) designed to isolate functional modifications from background residue features.

### Statistical Contingency Pipeline
The 25-module evaluation framework (M1-M25 in `PTM_Interpretability_Implementation_Plan_V3.md`) measuring latent-to-PTM association via 2x2 contingency tables, fold enrichment, selectivity, Permutation Nulls ($N=1,000$), and Bonferroni Family-Wise Error Rate control.
