# Stage 0: TopK learning-rate probe

Written for Member 1 (architecture and training). It records why stage 0 was run, what it measured, what it shows and what to do next.
Design context: `docs/research/topk-sweep-design.md`. Spec: `sweeps/topk_lr_probe.yaml`. wandb group: `topk_lr_probe`.

## 1. Premise

The long TopK runs of stage 1 need a learning rate (LR) per width. The literature value cannot be copied:

- Gao et al. 2024 found the best LR falls roughly as 1/sqrt(width), and that the best LR for training to convergence is about 4x lower than
  the best for a fixed short budget.
- Their numbers were measured at a 131,072-token batch. Ours is 4,096, so the gradient noise differs and the transfer is unproven.

So stage 0 asks one narrow question: **at each width, which LR is stable and best for a short budget?** The long runs then use about a quarter of it
(the converged-optimum ratio), an assumption that H4 (reference cell at 2x LR) checks later. Stage 0 is a nuisance-parameter calibration, not a
test of any thesis hypothesis (H0 to H3, H5 concern collapse, width, k and AuxK).

**Protocol.** TopK SAE with k = 32, no AuxK, ESM-2 650M layer 24 activations (`discovery_train`), batch 4,096 tokens, 3,000 steps
(about 12.3M tokens, roughly 2.4 passes over the ~5.2M training tokens), evaluation on `discovery_val`, collapse canaries off, one seed.

| width | LRs tried |
|---|---|
| 4,096 | 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3 |
| 10,240 | 1e-4, 2e-4, 4e-4, 8e-4 |

Run on Kaggle (2x T4) through `notebooks/train_sae.ipynb`, up to four runs at a time. All nine runs finished.

## 2. Results

Final-evaluation metrics from wandb (`val/*`). EV is explained variance on `discovery_val`; "never fired" is the fraction of latents that did not
activate on any validation token; c_dec is the mean pairwise decoder cosine.

| run | LR | val MSE | EV | never fired | c_dec |
|---|---|---|---|---|---|
| w4096 | 1e-4 | 21.44 | 55.7% | 0 | 0.0259 |
| w4096 | 2e-4 | 20.43 | 57.8% | 0 | 0.0252 |
| **w4096** | **4e-4** | **20.09** | **58.5%** | 0 | 0.0248 |
| w4096 | 8e-4 | 20.40 | 57.8% | 0.02% | 0.0238 |
| w4096 | 1.6e-3 | 21.54 | 55.5% | 0.07% | 0.0238 |
| w10240 | 1e-4 | 19.44 | 59.8% | 0 | 0.0254 |
| w10240 | 2e-4 | 18.26 | 62.3% | 0 | 0.0254 |
| **w10240** | **4e-4** | **17.88** | **63.1%** | 0 | 0.0249 |
| w10240 | 8e-4 | 18.19 | 62.4% | 0.05% | 0.0244 |

Validation EV (%) at each evaluation (every 500 steps), so the shape of each run is visible, not only its end:

| run | LR | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| w4096 | 1e-4 | 39.9 | 47.0 | 51.1 | 53.5 | 54.9 | 55.7 |
| w4096 | 2e-4 | 46.8 | 52.0 | 55.1 | 57.1 | 57.5 | 57.8 |
| w4096 | 4e-4 | 51.5 | 54.8 | 56.8 | 58.7 | 58.3 | 58.5 |
| w4096 | 8e-4 | 53.9 | 55.1 | 56.9 | 58.7 | 57.7 | 57.8 |
| w4096 | 1.6e-3 | 54.1 | 53.9 | 55.0 | 57.1 | 55.9 | 55.5 |
| w10240 | 1e-4 | 43.2 | 50.7 | 55.0 | 57.7 | 59.1 | 59.8 |
| w10240 | 2e-4 | 50.4 | 56.2 | 59.2 | 61.6 | 61.9 | 62.3 |
| w10240 | 4e-4 | 55.3 | 59.2 | 61.2 | 63.6 | 63.1 | 63.1 |
| w10240 | 8e-4 | 57.9 | 59.8 | 61.3 | 63.7 | 62.4 | 62.4 |

## 3. What it shows

0. **Why 3,000 steps, and is it enough?** It was a cost choice: nine runs for about 20 GPU-minutes, not a derived budget. It is enough to show that
   k = 32 is the limit (below), and not enough to rank learning rates fairly or to claim convergence.
1. **Learning rates from 2e-4 to 8e-4 reach the same plateau; 1.6e-3 is worse; 1e-4 is still climbing.** At LR >= 2e-4 EV stops improving by step
   about 2,000 (plateau near 58.7% at width 4,096 and 63.7% at width 10,240), so more steps would not lift it: the plateau is set by k and width.
   At 1e-4 EV is still gaining about 0.8 points per 500 steps, so that run is budget-limited and its low rank is partly an artefact of the short
   budget. The high-LR runs also dip after their peak (8e-4: 58.7 to 57.8; 1.6e-3: 57.1 to 55.5), which is constant-LR noise without weight
   averaging, and ranking on the final value penalises them. Evaluation noise is about 0.5 EV points, so 4e-4 versus 8e-4 is not resolved.
   No LR diverged (no NaN).
2. **The 1/sqrt(width) rule is not visible.** Width grew 2.5x, so the rule predicts a best LR about 1.6x lower at width 10,240. The measured optimum
   did not move on a 2x grid. The grid is too coarse and the runs are single-seed, so this is "not detected", not "refuted".
3. **Wider is better, by a modest amount.** At matched k and LR, width 10,240 gains about 5 points of EV over 4,096 (63.1% vs 58.5%).
4. **The EV gate is far away at k = 32.** The best of nine runs is 63.1%, against the 85% gate. LR cannot close that gap: the whole LR curve spans
   about 3 points. The curves show the plateau is reached by about step 2,000 at LR >= 2e-4, so the cause is k (and width), not training length. This is H3's open question.
5. **High LR shows up as dead latents.** Never-fired latents appear only at 8e-4 and above (0.02% to 0.07%), tiny but monotone with LR, and absent
   at or below 4e-4. This is the stability ceiling the probe was meant to find: it starts near 8e-4, and 1.6e-3 also loses EV.
6. **c_dec carries almost no signal here.** It drifts down slightly with LR (0.0259 to 0.0238) and is similar across widths. It becomes informative
   only when k varies (H2 predicts a valley over k), so it was not expected to discriminate in this stage.

## 4. Caveats

- One seed per cell. The differences among 2e-4, 4e-4 and 8e-4 (under 1 EV point) are inside the roughly 0.5-point evaluation noise, so the
  claim is "2e-4 to 8e-4 is the right region", not "4e-4 is exactly optimal".
- A 3,000-step budget favours larger LRs than a converged run would (the 1e-4 run is still climbing). The "quarter of it" rule for long runs is carried
  over from the literature and is still unverified at our batch size.
- Curves are evaluated every 500 steps on the validation set only, so the per-run noise (about 0.5 EV points) is a rough read, not a measured seed spread.
- Validation proteins are shorter than training proteins (mean length 224 vs 571), so absolute EV values are not comparable with other papers.
- Collapse canaries were off, so nothing here says anything about residue collapse.

## 5. What to explore next

| Priority | Question | Why the insights point there | How |
|---|---|---|---|
| 1 | Where does EV reach the 85% gate as k grows? | EV is far below the gate and insensitive to LR and width; k is the remaining lever | Stage 0.5 (`sweeps/topk_k_probe.yaml`, k = 64 to 1,024 at width 4,096, LR 4e-4, 3,000 steps); the stage-1 k range comes from it |
| 2 | Is EV still improving at step 3,000? | Answered from the curves: not at LR >= 2e-4 (plateau by about step 2,000), so low EV at k = 32 is a k and width limit | In the k-probe, compare EV at 2,000 vs 3,000 steps for each k and extend any k whose curve is still rising |
| 3 | Which LR for the long runs? | 2e-4 to 8e-4 are indistinguishable at both widths, so one value may serve both | Use 1e-4 to 2e-4 (about a quarter to a half of the short-budget region) for stage 1; the 2x-LR run (H4) is the check |
| 4 | Is a finer LR grid worth it? | Differences near the optimum are within plausible noise | No: skip, per the decision not to chase small gaps; seed spread from the reference-cell seeds in stage 1 gives the noise scale |
| 5 | Does AuxK change anything? | Dead latents appear at high LR even at k = 32; at larger widths and small k they will be worse | Stage 1 AuxK-on pairs (H5), unchanged |
| 6 | Throughput | Stage 1 cost depends on a CPU-bound per-row data loader (about 34k tok/s per run, 60 to 63k aggregate) | Done: the dataset now shuffles and cuts batches as tensors (`batch_size` mode of `ActivationPartitionDataset`) and converts fp16 to fp32 after the shuffle. On a synthetic 1,280-dim shard set the loader went from about 41 ms to about 18 to 21 ms per 4,096-row batch (2.0 to 2.3x) on a local PC. The k-probe note records the real Kaggle gain once measured |

## 6. Decisions this stage fixes

- LR for the k-probe: 4e-4 (inside the plateau region; no change to the spec).
- LR for stage-1 long runs: 1e-4 to 2e-4 at both widths, to be revisited after the k-probe and checked by H4.
- Reporting rule: every sweep summary shows the EV curve (or the last two evaluations) beside the final value; runs are not ranked on the final value alone.
- Stability ceiling: avoid LR at or above 8e-4 at these widths.
