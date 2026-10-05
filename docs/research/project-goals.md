# Project goals, targets and evidence so far

Written for Member 1 (architecture and training) and Member 2 (evaluation), to agree on what the thesis is aiming at and how it will be judged. Status as of 2026-10-06. Companion to `docs/research/topk-sweep-design.md`.

## 1. The question and what we will claim

Do pretrained protein language models (ESM-2 650M, layer 24) emergently encode post-translational modifications (PTMs), and can sparse autoencoders disentangle them without residue collapse?

The primary claim is stated relative to the best monolithic baseline, not as an absolute threshold: *Arm D yields significantly more within-stratum PTM-selective latents than the best monolithic SAE at the same L0, against the same controls.* A null result ("collapse is inherent, modular heads do not fix it") is reportable under the same protocol.

## 2. Goals ladder

Each rung produces something publishable on its own, so a failure higher up does not undo the rungs below.

### Short term: the thesis

| # | Goal | Done when |
|---|---|---|
| S1 | Baseline frontier: EV-versus-L0 for TopK, then JumpReLU, BatchTopK, Gated | Curves with seed spread; reference cells chosen |
| S2 | Controls: random-baseline SAE, linear-probe ceiling, sequence-window baseline, raw-neuron baseline | Each yields a number on `discovery_val`. Done early, as a go/no-go for everything after |
| S3 | The measurement: within-stratum PTM selectivity, redefined collapse rate, InterPLM-style F1, on Arms A to C | One scorecard for all arms (with Member 2) |
| S4 | Modular candidates: Stratified and Matryoshka comparator first; PolySAE only after crosstalk is defined | Each compared at matched L0 against the best monolithic |
| S5 | Arm D, unsupervised; guided Arm D as an extension | Held-out evaluated once |

### Medium term: a benchmark

Release the corpus, cluster-level split, ambiguity masks and negative tiers; the scoring protocol; Arms A to D dictionaries at several sparsity levels; and a catalog of latents that survive the permutation null. SAE features are not unique across seeds, so the standard is the benchmark and the catalog, not one set of weights.

### Long term

Reference dictionary suite across layers and model scales; a real model of crosstalk across residues (not single-token); more PTM types and organisms; hypothesis generation with wet-lab validation; steering for protein design (AxBench suggests the causal results may be weak, so plan for a null).

## 3. Targets, by role

**Gates (eligibility, not results).** EV within a few points of the monolithic frontier at the same L0, not an absolute 85%; dead latents under 5% with the window stated; train/validation gap reported; L0 always reported. Downstream loss recovered (ESM-2 masked-LM loss with the reconstruction spliced in) is a better fidelity check than EV and is worth adding.

**Outcomes (the claim).**
1. Within-stratum PTM selectivity per PTM type and stratum: best-latent AUPRC and fold-enrichment against the stratum-preserving permutation null with Bonferroni; the number of surviving latents per PTM type.
2. Collapse rate redefined as the fraction of PTM-associated latents not selective within their stratum, with a confidence interval from a cluster-level bootstrap.
3. Linear-probe ceiling per PTM and stratum on raw layer-24 activations (does ESM-2 encode it at all), and the ratio of best latent to probe.
4. A latent must beat the best single raw dimension.
5. InterPLM-style precision/recall/F1 against annotations, so published Arms A to C and Arm D share a metric.

**Controls (what makes a positive result believable).** SAE on random-weight ESM-2 activations or with random decoder directions; a sequence-only model over a plus or minus 7 residue window per site (does ESM-2 encode more than local motif); hard-negative shortcut check.

**Protocol rules, fixed before results.** Report the full EV-versus-L0 curve and compare at matched L0 (matched EV is not a field standard); at least 3 seeds, and a difference counts only if larger than twice the seed range; select everything on `discovery_val`; evaluate on `held_out` once; one scorecard for all arms.

## 4. Evidence so far

**Activation geometry** (`scripts/diagnose_activations.py`, 100k-token train and val samples, 4 shards each, so a first look):

| | Train | Val |
|---|---|---|
| Top-5 dimension variance share | 5.6% | 6.3% |
| Components for 80% / 95% variance | 628 / 1,048 | 491 / 971 |
| Participation ratio | 194 | 74 |
| Variance explained by amino acid alone | 2.3% | 2.5% |
| Chain-end variance share / token share | 2.1% / 2.8% | 1.9% / 2.7% |
| Ideal per-token sparse-PCA EV at k=32 / 64 / 512 | 36% / 47% / 92% | 48% / 58% / 94% |

No massive dimensions, a flat spectrum, and little variance explained by residue identity alone. The unexplained variance follows residue frequency. Val is more concentrated than train, so val EV and train EV are not comparable. Chunked reductions now let the script run at 200k tokens, with the same picture (80% variance needs 662 components, participation ratio 190, amino acid alone 2.3%, sparse-PCA EV 35% at k=32); a 300k-token attempt was killed without output, so 200k is the practical limit on this PC.

**k-probe** (TopK, width 4,096, lr 4e-4, 3,000 steps, val EV, no dead latents except 0.3% at k=1,024):

| k | 64 | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|
| EV | 62.4% | 67.5% | 74.3% | 84.9% | 97.0% |

The 85% gate is reached only near k=512, which is not sparse. Together with the geometry above, the absolute 85% gate cannot be met by a sparse code on this layer, which is why the gate is relative.

**Multi-PTM residues** (from `labels_stratified.parquet`, 498,619 label rows on 418,032 labelled residues; 49,495 residues carry more than one PTM type). They are almost all lysine:

| Partition | Labelled K residues | K with more than one type | Share | Labelled ST residues | ST multi |
|---|---|---|---|---|---|
| discovery_train | 83,891 | 30,092 | 36% | 164,110 | 3,963 |
| discovery_val | 11,670 | 4,342 | 37% | 23,315 | 588 |
| held_out | 26,992 | 8,785 | 33% | 46,748 | 1,240 |

Other strata have almost none (N: 1, Y: 19, R: 9 in `discovery_train`). The commonest pairs on one residue are acetylation with ubiquitination (16,850 in `discovery_train`), sumoylation with ubiquitination (12,763) and acetylation with sumoylation (6,083); O-GlcNAcylation with phosphorylation on S/T has about 3,000 overall. So the same-residue crosstalk test has ample data on lysine and almost none elsewhere. The labels already carry a `crosstalk` flag. A caution for control matching: heavily studied proteins collect modifications of every type, so single-modified controls should come from the same proteins.

**Literature support.**

| Target | Support |
|---|---|
| Probe and raw baselines | Strong: [Kantamneni et al. 2025](https://proceedings.mlr.press/v267/kantamneni25a.html) (SAEs rarely beat baselines at probing), [AxBench](https://arxiv.org/pdf/2501.17148) (SAEs not competitive for concept detection, weak at steering) |
| Random-baseline controls | Strong: [Heap et al.](https://arxiv.org/html/2501.17727v2), [Sanity Checks for SAEs](https://arxiv.org/html/2602.14111v1) (random baselines match trained SAEs on several metrics) |
| Concept scoring against annotations | [InterPLM](https://arxiv.org/pdf/2412.12101) uses F1 against Swiss-Prot concepts; our contingency/permutation design is stricter but is our own |
| Compare across L0; Matryoshka comparator; EV is a poor proxy | [SAEBench](https://www.neuronpedia.org/sae-bench/info), [SynthSAEBench](https://www.alphaxiv.org/abs/2602.14687.md) |
| Collapse thresholds (25%, 2x), twice-the-seed-range rule | Our own heuristics, no source |

## 5. Decisions taken

- Collapse is judged by within-stratum enrichment for every model; the label-free residue-dominance gate stays a diagnostic for monolithic models only, because stratified heads would fail it by construction.
- Arm D is trained unsupervised; a label-guided arm is a separate extension.
- The primary claim is relative to the best monolithic baseline; the linear-probe ceiling is a core target; random-baseline controls, the sequence-window baseline and InterPLM-style F1 are added to scope.
- S2 runs right after S1, before modular architecture work.
- Comparisons are made at matched L0 of 64 and 256, plus the full EV-versus-L0 curve for every model. 64 is the published InterProt setting (Arm A is comparable); 256 sits mid-curve (about 74% EV) where reconstruction still has headroom.
- Capacity is a second budget. Every Arm D variant is compared with a monolithic TopK of the same total width and the same L0 (so width 10,240 stays in the baseline grid). Published Arms A to C (4,096 and 10,240) are reference points, not matched comparisons. Parameter counts are reported for every model; if Arm D carries many extra parameters (interaction tensors, routing, cascades), a parameter-matched wider monolithic is added as a sanity control.
- Crosstalk means same-residue multi-PTM propensity first (several modifications competing for one residue). ESM-2 reads the unmodified sequence, so its latents can only encode propensity and context, never modification state; the thesis says so explicitly. Before PolySAE is built, a go/no-go checks whether pairwise products of a trained SAE's active latents predict its reconstruction residual better than chance on multi-modified tokens (cluster-level cross-validation, against matched single-modified controls). PolySAE stays on the list only if that finds signal. The neighbouring-residue meaning (priming) is a cheap optional add-on: mutate a serine to Asp/Glu and measure how nearby lysine latents shift.

## 6. Open questions

- **Crosstalk go/no-go.** Which trained SAE to run it on (k=256 is the proposal), and how to match single-modified controls to multi-modified tokens.
- **Stratified heads.** Data per head (K, ST, N, Y are a small share of about 5.2M training tokens) and which head handles the roughly 70% of other residues.
- **Modular budget definition.** How L0 and width are counted for a stratified or other modular model: active latents across all heads per token with the sum of head widths matched to the monolithic width (the working proposal), or per head. Precedent: [Switch SAEs](https://arxiv.org/pdf/2410.08201) route each input to one small expert SAE and are compared under FLOP-matched and width-matched conditions. A stratified SAE is the same idea with a hard-coded router (the amino acid) instead of a learned one, so report compute (FLOPs) as well. Needs a decision before the stratified model is designed.
- **Validation sample.** Val proteins are shorter and more concentrated than train; confirm what that does to EV and to the selectivity metrics.
- **Open items from the TopK audit:** dead-latent window (500k tokens against 10M in Gao et al.), shuffle buffer, no weight EMA, and the token-count mismatch between configs and docs.
