# Stage 0.5: TopK explained-variance-versus-k probe

Written for Member 1 (architecture and training). It records why stage 0.5 was run, what it measured, what it shows and what to do next.
Design context: `docs/research/topk-sweep-design.md`; previous stage: `docs/research/topk-stage0-lr-probe.md`; goals: `docs/research/project-goals.md`.
Spec: `sweeps/topk_k_probe.yaml`. Notebook: `notebooks/train_sae_k_probe.ipynb`. wandb group: `topk_k_probe`.

## 1. Premise

Stage 0 found that k = 32 at width 4,096 plateaus near 58% explained variance (EV) on `discovery_val`, far below the 85% gate, and that learning rate moves EV by only about 3 points. Stage 0.5 asks one narrow question: **where, as k grows, does EV reach the gate, and what does the curve look like?** The answer fixes the k range of stage 1 from data instead of from convention.

**Protocol.** TopK SAE, width 4,096, learning rate 4e-4, no AuxK, ESM-2 650M layer-24 activations (`discovery_train`), batch 4,096 tokens, 3,000 steps (about 12.3M tokens, roughly 2.4 passes over the ~5.2M training tokens), evaluation on `discovery_val` every 500 steps, collapse canaries off, one seed. k in {64, 128, 256, 512, 1,024}.

Run on a Kaggle kernel with two T4s, pushed through the API with the private secrets dataset (this was the first run to use it: the clone, the Weights & Biases logging and the data sync all worked). The first four runs ran at once; k = 1,024 followed alone. All five finished.

## 2. Results

Final-evaluation metrics (`val/*`). "Never fired" is the fraction of latents that did not activate on any validation token; c_dec is the mean pairwise decoder cosine; the k = 32 row is from stage 0 (same width and learning rate).

| k | val MSE | EV | cosine mean / p10 | never fired | c_dec |
|---|---|---|---|---|---|
| 32 (stage 0) | 20.09 | 58.5% | | 0 | 0.0248 |
| 64 | 18.22 | 62.4% | 0.917 / 0.871 | 0 | 0.0236 |
| 128 | 15.74 | 67.5% | 0.929 / 0.891 | 0 | 0.0220 |
| 256 | 12.43 | 74.3% | 0.944 / 0.915 | 0 | 0.0209 |
| 512 | 7.30 | **84.9%** | 0.968 / 0.952 | 0 | 0.0224 |
| 1,024 | 1.47 | 97.0% | 0.994 / 0.991 | 0.34% | 0.0209 |

Validation EV (%) at each evaluation, so the shape of each run is visible, not only its end:

| k | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| 64 | 55.2 | 58.4 | 60.5 | 62.0 | 62.2 | 62.4 |
| 128 | 60.8 | 63.6 | 65.6 | 67.0 | 67.4 | 67.5 |
| 256 | 68.3 | 70.8 | 72.5 | 73.7 | 74.1 | 74.3 |
| 512 | 77.4 | 81.0 | 83.1 | 84.2 | 84.7 | 84.9 |
| 1,024 | 91.2 | 95.8 | 96.4 | 96.7 | 96.8 | 97.0 |

Cost: the four concurrent runs took about 954 s each (about 17,900 tokens/s per run, about 72,000 aggregate, input-pipeline bound as in stage 0); k = 1,024 alone took 488 s.

## 3. What it shows

1. **The 85% gate is reached only at k of about 512.** That is 40% of the input dimension (1,280) and 12.5% of the dictionary. The ratio is not sparse in any usual sense, so the absolute gate cannot be met by a sparse code on this layer. This agrees with the activation diagnosis (`scripts/diagnose_activations.py`): the variance spectrum is flat (about 660 principal components for 80%), with no massive dimensions to remove.
2. **EV gains per doubling of k grow, they do not shrink.** From k = 32 upward the gains are +3.9, +5.1, +6.8, +10.6 and +12.1 points. A flat spectrum behaves this way: each extra atom explains about the same amount, so coverage grows roughly linearly in k, and there is no knee at which a small k suffices. Choosing k is therefore a choice of where on a smooth curve to sit, which supports comparing architectures along the whole curve (or at matched L0) instead of at one gate.
3. **The runs are converged.** Between steps 2,500 and 3,000 EV moves by 0.2 points or less at every k, so more steps at this learning rate would not close the gap to 85%: the plateau is set by k and width, as in stage 0. (The curves say nothing about feature quality, which may need longer training than EV does.)
4. **Dead latents are not the problem at any k.** None at k up to 512, 0.34% never fired at k = 1,024. AuxK is not needed for these runs; it matters only at small k or large width.
5. **c_dec has no valley.** It stays between 0.0209 and 0.0236 with no trend, a spread near the noise level. Hypothesis H2 (a valley in the decoder cosine over k) is not supported here, though the measurement is coarse; it may become informative once collapse is measured.
6. **Reading k = 1,024.** At EV 97% and cosine 0.994 the code is close to a dense re-encoding of the input (80% of the input dimension, 25% of the dictionary active); it is a ceiling for the curve, not a candidate baseline.

## 4. Caveats

- One seed per cell and a single width, so differences of a point or two between neighbouring k are not resolved.
- A 3,000-step budget, 2.4 passes over the training tokens: converged for EV, not shown to be converged for features.
- Collapse canaries were off, so nothing here says anything about residue collapse.
- The k-sparse PCA reference from the diagnosis (about 48%, 58%, 70%, 83% and 94% at k = 32 to 512 on a small validation sample) is not directly comparable: it picks coefficients by magnitude with either sign, which doubles the effective atoms, and it was measured on a different sample. It is a loose guide, not a target. It does warn that at k of 128 and above this SAE sits below an exact orthogonal-basis reference, so a wider or longer-trained dictionary could still improve the high-k end.
- Validation proteins are shorter and more concentrated than training proteins, so absolute EV values are not comparable with other papers or with train EV.

## 5. What to explore next

| Priority | Question | Why | How |
|---|---|---|---|
| 1 | What does residue collapse do as k grows? | EV is only a gate; the thesis measures collapse, which no run has yet reported | The same k grid with the canaries on, plus extra seeds at one k for the noise level |
| 2 | Does width change the curve? | Stage 0 showed +5 points at k = 32; H1 predicts less collapse at width 10,240 | A second width at k of 64, 256 and 512 |
| 3 | Are long runs different from 3,000 steps? | EV converges early, features may not | One k at about 10,000 steps |
| 4 | How do JumpReLU, BatchTopK and Gated sit on the same curve? | Phase 2 compares sparsity mechanisms at matched L0 | Each needs its own sparsity-coefficient sweep to land on target L0 values |
| 5 | Controls (random baseline, probe ceiling, sequence-window baseline) | They decide whether any collapse number means something | See `docs/research/project-goals.md`, goal S2 |

## 6. Decisions this stage fixes

- The absolute 85% EV gate is dropped as a selector: it is reachable only at k of about 512. EV stays a control, reported along the whole curve and compared at matched L0.
- Stage 1 uses k in {32, 64, 128, 256, 512} (k = 16 and 1,024 add nothing: the first is far below any gate, the second is near dense).
- The learning rate stays 4e-4 for short runs; the long-run value from stage 0 (1e-4 to 2e-4) is still to be checked.
- Reporting rule from stage 0 stands: show the EV curve beside the final value.
