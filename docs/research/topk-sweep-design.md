# TopK baseline sweep: literature, audit and design

Written for Member 1 (architecture and training), to decide and defend the Phase 2 TopK settings.

## 1. Why this exists

The first TopK sweep (commit 91444c5: k in {16, 32, 64} x learning rate in {4e-4, 1e-3}, width 4,096) had no recorded basis and no
way to read its result. The plan documents state one hypothesis only, an attribution question
(`docs/thesis-meeting-phase2-report.md`: is residue collapse "an inherent property of the PLM's representations or simply an artifact
of the mathematical sparsity formulation"?), and no falsifiable prediction about k, width or learning rate. This document supplies
them, grounded in the literature, and fixes the measurements first.

## 2. What the literature says

| Source | Finding | What it means here |
|---|---|---|
| [Gao et al. 2024](https://arxiv.org/abs/2406.04093) (TopK) | Plain Adam, constant LR, eps 6.25e-10, EMA weights, AuxK (alpha 1/32, k_aux about d/2); the best LR falls about 1/sqrt(width) and the best LR for training to convergence is about 4x lower than for a fixed budget; reconstruction improves slowly with k | Our core matches; eps, input scale and AuxK checked below |
| [InterPLM](https://arxiv.org/html/2412.12101v1), [InterProt](https://proceedings.mlr.press/v267/adams25a.html) | SAEs on ESM-2 find thousands of interpretable features per layer, including phosphorylated residues; neurons are in superposition; wider dictionaries give more family-specific features | SAEs can surface PTM features; width matters |
| [PTM-Mamba](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12074982/) and ESM-2 probing | PLMs encode modification-prone residue context (enough to predict sites) but "have yet to represent" modification state | Expect "modifiable context" features: collapse may be partly inherent |
| [Feature hedging](https://arxiv.org/html/2505.11756v1) | A narrow SAE merges components of correlated features, worse the narrower it is | Residue identity and PTM state are correlated: narrow dictionaries should collapse more |
| [Sparse but Wrong](https://arxiv.org/html/2508.16560v3) | L0 too low mixes correlated features, too high gives degenerate ones; most common SAEs have L0 too low; the mean pairwise decoder cosine (c_dec) has a valley at the right L0; sweep L0 broadly | k = 32 / 64 may be too low: sweep k widely; EV alone is a misleading selector |
| [Matryoshka SAEs](https://arxiv.org/html/2503.17547v1) | Nested dictionaries cut absorption (0.49 to 0.05) for about 2% lower EV and +50% training time | The natural hierarchy-aware comparator for Phase 3 (not for this baseline) |
| [SAEBench](https://arxiv.org/html/2503.09532v3) | Reconstruction fidelity is a poor proxy for feature quality | Use EV as a gate only |
| [ProtSAE](https://arxiv.org/pdf/2509.05309) | Annotation-guided SAEs improve biological interpretability | An option for the guided Arm D, not the unsupervised baseline |

## 3. Audit of our implementation

**Matches Gao et al.:** decoder init and unit-norm renormalisation after every step, transposed-decoder encoder init, geometric-median
b_dec, gradient projection and step order, ReLU-then-top-k, AuxK mechanics (mask, detached residual, k_aux = 512 at d_in = 1280),
constant-LR plain Adam (AdamW with zero weight decay), explained variance as 1 - SSE / total variance. Extraction takes the output of
block index 24 (`hidden_states[25]`, pre-final-layernorm), drops the cls/eos/pad positions, and held_out is never loaded.

**Fixed with this design (all with tests):**

- TopK inputs are rescaled to E||x||^2 = 1 like JumpReLU and Gated (`activation_scale`, also applied to the AuxK loss); the raw activation
  scale (norm quantiles, variance share of the top 5 dimensions, largest mean/std) is logged once as `activation/*`.
- Adam eps and betas are explicit (6.25e-10, (0.9, 0.999)).
- Explained variance accumulates in float64 (a massive-activation dimension broke `E[x^2] - mean^2` in float32).
- `val/dead_latent_fraction` is the TRAINING census (500k-token window) under a `val/` name; the eval-time number is now
  `val/never_fired_fraction`. `val/decoder_pairwise_cosine` (c_dec) is logged at every eval.
- Residue dominance: only positive activations count, a latent is scored once it has a full top-k of them, and the primary number is
  `residue_dominance/collapse_rate_alive` with the rate at thresholds 0.5 / 0.9 and `chance_rate` (independent draws from the observed
  amino-acid frequencies). The old `collapse_rate` (all-latents denominator) is kept, but dead latents deflated it.
- PTM-concentration ratios also report `*_supported` versions that require at least 20 active tokens per stratum (3 on modified residues).
- A manifest covering less than `min_partition_coverage` (default 0.99) of a partition fails at start; the 8M pilot configs set 0.
- The sweep summary reads each run's `metrics.jsonl` instead of console text.

**Still open (documented, not fixed):** shuffling is a 65,536-token buffer over consecutive proteins (about 115), so batches are not i.i.d.;
no EMA of weights (Gao uses 0.999); the dead window is 500k tokens versus Gao's 10M, so dead fractions are not comparable to theirs and
are biased by k/width; for TopK, "L0" is the mean number of active latents (at most k); validation proteins are shorter than training
ones (mean length 224 vs 571); the train-token figures in configs (5.21M) and docs (5.08M) disagree and should be reconciled from the
real manifest; checkpoints trained before this change are not comparable (input scale).

## 4. Hypotheses (fixed before any result)

| | Hypothesis | Prediction | Refuted / no effect if |
|---|---|---|---|
| H0 | Collapse is an artifact of the sparsity setting (thesis question) | Some feasible cell (EV >= 85%, dead < 5%) has an alive collapse rate below the documented 25% target | No feasible cell reaches < 25% and nothing moves with k or width: collapse looks inherent (within TopK) |
| H1 | Width reduces collapse (hedging; InterProt) | rate(10,240) < rate(4,096) at matched k | Difference within the seed spread, or reversed |
| H2 | L0 has a valley: low k mixes (more collapse, higher c_dec), very high k degenerates | collapse and c_dec fall, then flatten or rise, over k | Flat across k |
| H3 | The fidelity gate is reachable | EV >= 85% at some k at each width | Never reached |
| H4 | Learning rate and steps are nuisance | The reference cell at 2x LR gives the same collapse | The result moves with LR |
| H5 | Dead latents do not drive the conclusion | At each AuxK-on pair, dead fraction falls (to < 5%) while collapse and c_dec stay within the seed spread of the AuxK-off run | They change by more than the spread: AuxK-on becomes the baseline |

*Real* means larger than twice the range across 3 seeds of the reference cell. All hypotheses concern TopK only; a TopK-only result cannot
fully separate "inherent" from "TopK-specific". Deferred: JumpReLU (needs a sparsity-coefficient sweep to match L0), Matryoshka, guided SAEs,
batch size, layer.

## 5. Measures

Primary: `residue_dominance/collapse_rate_alive` at the final evaluation, with its stability over the last 20% of training, plus the
0.5 / 0.9 sensitivity rates and `chance_rate`. Mixing: `val/decoder_pairwise_cosine`. Secondary: per-stratum (K, ST, N) supported
concentration metrics, per active latent. Gates and controls: `val/explained_variance`, `val/never_fired_fraction`, `train` dead fraction,
mean L0. Collapse and dominance are measured on the final model, not on `best/`.

## 6. Design

- **Stage 0, learning-rate probe** (`sweeps/topk_lr_probe.yaml`): 9 runs of 3,000 steps at k = 32, canaries off; width 4,096 at
  1e-4 to 1.6e-3, width 10,240 at 1e-4 to 8e-4. Purpose: the stable ceiling and best short-budget LR per width, because the literature LR was
  measured at a 131,072-token batch (here 4,096). The long runs use a quarter of it (Gao's converged-optimum ratio, checked by H4).
- **Stage 0.5, explained-variance-versus-k probe** (`sweeps/topk_k_probe.yaml`, run by `notebooks/train_sae_k_probe.ipynb` on the same Kaggle
  slug): added after stage 0 showed that k = 32 at width 4,096 plateaus near 58% explained variance, far below the 85% gate. Five 3,000-step runs
  at k = 64, 128, 256, 512, 1,024 (width 4,096, learning rate 4e-4, canaries off) show where the gate is reached, so the stage-1 k range is
  chosen from data. Real Kaggle throughput measured during stage 0 (about 34k tok/s per run, 60-63k aggregate on two T4s, CPU-bound by the
  data loader) sets the cost of everything after it. The loader has since been made batch-level (tensor shuffle, fp32 conversion after the
  shuffle): about 2x faster on its own in a local benchmark, so the figures above are the old, slower loader.
- **Stage 1, dose-response grid** (`sweeps/topk_grid.yaml`, written after stages 0 and 0.5): width 4,096 at k in {16, 32, 64, 128, 256, 512}; width 10,240
  at k in {64, 128, 256}; 2 extra seeds at the reference cell (k = 64, width 4,096); that cell at 2x LR (H4); AuxK-on (`auxk_coefficient: 0.03125`)
  at width 4,096 with k = 16, 64, 256 and at width 10,240 with k = 64 (H5): 16 runs; 24,000 steps at width 4,096 and about 41,000 at 10,240
  (tokens to convergence grow about width^0.6); canaries on.
- **Stage 2:** 2 more seeds only for contrasts that stay ambiguous against the seed spread.
- **Selection:** none up front; every cell is reported and analysed together.

## 7. Reading the result

No feasible cell below 25% and no k/width effect beyond the spread: collapse is robust to the sparsity setting (supports "inherent",
motivates the modular Phase 3). Some feasible cell below 25%: the artifact reading; that cell is the baseline Phase 3 must beat. Effects
present but never below target: mitigated, not removed. The thresholds (top-10, 0.7, 2x baseline) are heuristics; anything stronger needs
Member 2's permutation-null procedure.

## 8. Cost and risks

Stage 0 about 20 GPU-minutes; stage 1 about 8 GPU-hours sequential-equivalent (k barely changes step time; AuxK adds about 10%), run
several at once by the launcher. About 5.2M unique training tokens means 19 epochs at 24,000 steps and about 32 at 41,000: watch the
train/validation gap. EV >= 85% may fail at small k (H3).
