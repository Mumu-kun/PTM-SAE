# Train / Val / Held-out Split Strategy: Talking Points

**Project**: PTM-SAE Engine | **Author**: Member 1 | **Date**: October 2026
**Corpus**: 18,128 reviewed human proteins, 11,909 CD-HIT@40% clusters, 498,619 labelled sites

---

## What it does

- Proteins stay grouped in CD-HIT@40% **homology clusters**; a cluster is never split across partitions.
- One optimiser assigns whole clusters to `discovery_train` / `discovery_val` / `held_out` (70/10/20% of tokens) so the **three partitions look alike** on: tokens, sites per PTM type, residues per chemical stratum, cluster-size mix, proteins carrying a site, crosstalk sites.
- A **cd-hit-2d audit** across each boundary merges any cluster pair still above 40% identity and re-solves.
- The whole assignment, solver settings and achieved-vs-target table are stored in `split_manifest.json` (deterministic, reproducible).

## Why

- The old split balanced only totals. Measured on the real corpus, **val was not like train**: 94% of its tokens sat in singleton clusters (train 44%), its proteins averaged 196 aa (train 483 aa), and 17 of 37 balance features were more than 25% off target.
- That matters for the thesis: val loss and the Residue-Collapse canary run on val, and V12b's held-out re-estimate needs a population comparable to discovery.
- The first, pre-Parquet split never clustered at all (cluster count = protein count), so it had no homology protection.
- CD-HIT compares members only to a cluster representative, so >40% pairs can exist *between* clusters; nothing checked this.

## Before / after (measured)

| | Old | New |
|---|---|---|
| Val: mean / max deviation from target | 30.9% / 100% | **0.08% / 0.49%** |
| Held-out: mean / max deviation | 11.3% / 34.0% | **0.04% / 0.61%** |
| Features > 25% off target (val) | 17 of 37 | **0** |
| Token share train / val / held-out | 68.8 / 9.8 / 21.4% | **70.0 / 10.0 / 20.0%** |
| Singleton-cluster token share | 44 / 94 / 49% | **50 / 50 / 50%** |
| Mean protein length (aa) | 483 / 196 / 426 | **409 / 414 / 425** |
| Val proteins with >= 1 site | 85.6% | **94.5%** |

Method: MILP (HiGHS) places the 150 largest clusters exactly, the long tail is rounded deterministically, then a local search polishes. About 3 minutes, under 3 GB RAM, identical output on re-run. Per-type floors (>= 200 sites in held-out, >= 100 in val where reachable) are verified or the split fails.

## What it buys the thesis

- Val-based tuning and canaries measure the population the SAE actually trains on.
- Held-out V12b gets comparable sites for every eligible PTM type and exact enrichment denominators (residues per stratum are balanced explicitly).
- Every claim about the split is auditable from the manifest instead of assumed.

## Decisions and asks

1. **Held-out membership changes.** Only 20.6% of the old held-out proteins remain held-out; 5,725 proteins switch between discovery and held-out. Anything keyed to the N1 holdout (M6 holdout histograms, V12b) must be re-run. N1's assignment is kept as `split_n1`.
   - *Fallback if sign-off is a problem:* keep held-out exactly as is and split only discovery. Measured: val mean 0.11% / max 1.2% in 7 s. Held-out keeps its current 11% mean deviation (cluster-size skew).
2. **Activations:** 4,801 of 14,605 new discovery proteins have no 650M activations yet (coverage 67.4% train, 65.1% val). Re-extraction resumes from the existing manifest.
3. **Publish:** the corpus must be re-split on a machine with `cd-hit-2d` (audit) and republished to `ptm-sae-corpus`.

## Honest limits

- The cd-hit-2d audit is implemented and tested against faked tool output; **it has not yet run on the real corpus** (no CD-HIT on the dev machine).
- CD-HIT@40% misses remote homologs, and ESM-2's own pretraining already saw these proteins; neither is fixable by splitting.
- Nine ultra-rare or single-cluster-dominated PTM classes (eight pooled "other" classes plus neddylation) cannot be balanced and are excluded and reported.
- The solver stops at a node cap, not a proven optimum; the deterministic polish does the fine balancing.
