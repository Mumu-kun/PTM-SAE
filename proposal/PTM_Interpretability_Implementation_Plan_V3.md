# PTM Interpretability Pipeline — Implementation Plan
### Execution-ordered specification

**Purpose of this document.** This is the complete plan, reorganised from conceptual design order into **code execution order**. Every table and figure is emitted by the module that first has all its inputs — nothing is deferred to a final "reporting step." A coding agent should implement Part 0 first (configuration and invariants), then modules M1…M25 in sequence.

**Reading contract.** Each module states: **Inputs** · **Computation** · **Emits** · **Gate** (what must be true to proceed) · **Cost**. Rationale is given inline where a decision could otherwise look arbitrary — an agent should not need to consult another document to understand why a choice was made.

**Superseded documents.** This replaces `PTM_Interpretability_Plan_Steps_1_to_3.md`, `..._Steps_4_and_5.md`, and `..._Step_6.md`. Nothing from those files is omitted here.

**Sources referenced by abbreviation.** SPIRAL (Rahman group, BiRNA-BERT) · InterPLM (Simon & Zou, *Nature Methods* 2025) · InterProt (Adams et al., ICML 2025) · VF (Chon & Andreopoulos, β-lactamase validation framework) · VG&A (Villegas Garcia & Ansuini, ICLR 2025 Workshop) · Gujral (PNAS 2025) · ProtSAE (AAAI 2026) · ESMC-SAE (Hu et al. 2026) · ProtSyntax · **COMPASS-PTM** (Zhang, Cao et al., *Nature Communications* 17:6450, 2026) · H&F (Hunklinger & Ferruz, *Nat. Mach. Intell.* 2026) · SAEBench (Karvonen et al., ICML 2025) · P&B (Paulo & Belrose, ICLR 2026) · AxBench (ICML 2025).

---

# PIPELINE ACTION INDEX

Every action the pipeline performs, in execution order. No descriptions — find the stage name below, then
read its section for detail.

**Acquisition**
- Download reviewed human Swiss-Prot proteome with feature fields
- Download dbPTM 2025 site table
- Download CPLM 4.0 human species file (site data only, not the annotation layer)
- Download qPTM human site table (independent-curation replication set)
- Download O-GlcNAcAtlas 4.0 unambiguous and ambiguous datasets
- Download N-GlycositeAtlas site table
- Download N-GlyDE glycosylated and verified-non-glycosylated sequon sets
- Download ESM-2 650M and ESM-2 8M weights
- Download all 21 SAE checkpoints (unnormalised variants for the InterPLM arms)
- Write acquisition manifest with checksums and access dates

**Corpus construction**
- Filter to reviewed human proteins at or below 1,022 residues
- Count residues per stratum exactly (lysine, serine/threonine, asparagine, cysteine, arginine, tyrosine)
- Cluster the corpus at 50% sequence identity
- Assign every cluster to the discovery or held-out partition, stratified by PTM type
- Write the split manifest

**Label construction**
- Canonicalise PTM type names across all source databases
- Filter to experimental evidence tiers; drop homology-transfer and text-mining-only records
- Validate every site position against the reference sequence and the expected residue chemistry
- Verify CPLM flanking peptides against reference sequences
- Build the ambiguity exclusion mask from O-GlcNAcAtlas ambiguous sites and CPLM localisation-probability classes
- Deduplicate sites across sources and record source multiplicity
- Pool PTM types below the minimum site count into an "other" class
- Propagate cluster and partition labels to every site
- Build the three-tier negative sets (verified, hard, background)
- Build both corpus definitions (annotated-protein-restricted and proteome-wide)
- Flag crosstalk residues and record their type combinations
- Build all three label assignments (residue-stratified, naive all-residue, and qPTM replication)
- Emit dataset provenance table, label composition table, dataset composition figure

**Calibration**
- Load base models and all SAE checkpoints
- Scan the calibration protein set for per-latent maximum activation
- Derive normalisation constants and rescale encoder/decoder weights
- Construct histogram bin edges with threshold-sweep values pinned as exact boundaries
- Record the dead-latent census
- Run the timing extrapolation pass

**Fidelity gate**
- Compute explained variance, cosine similarity, KL divergence, percent loss recovered, and L-zero per arm per layer
- Report the normalisation overflow rate
- Estimate intrinsic dimension across all base-model layers
- Count attempted, alive, and analysable latents per arm
- Emit instrument validation table and figure
- **Gate: an arm failing fidelity is excluded from all downstream claims**

**Streaming accumulation pass**
- Run one base-model forward pass per sequence returning all hidden states
- Encode every hidden state through every SAE checkpoint
- Accumulate per-latent per-label activation histograms for all three label assignments and both partitions
- Accumulate exact sums, sums of squares, and firing counts
- Accumulate the per-latent amino-acid firing matrix and top-activation heap
- Accumulate per-protein mean activations
- Discard activations each batch
- **Gate: histogram row sums must match known label counts**

**Statistical testing**
- Build contingency tables at every threshold from histogram suffix sums
- Select chi-square or Fisher's exact per cell by expected-count rule
- Compute fold enrichment, selectivity, effective types claimed, and odds ratio over PTM types only
- Compute Mann-Whitney U and Cliff's delta conditional on firing
- Compute area under precision-recall and area under ROC
- Compute Welch's t and Cohen's d as secondary comparability metrics
- Compute the chemically-impossible-association fraction for the naive background
- Apply the activity pre-filter and record attempted versus analysed

**Permutation null**
- Shuffle labels within stratum, positives count held fixed, 1,000 times
- Recompute the full statistic vector per permutation
- Repeat for the naive background
- Repeat with the negative population restricted to the hard tier
- Repeat for the qPTM replication label set
- Flag cells failing the null-reliability edge conditions
- Derive per-cell percentile thresholds

**Correction and candidate selection**
- Apply Bonferroni per dictionary per stratum, and per all-types for the naive background
- Apply Benjamini-Hochberg and permutation false discovery rate as secondaries
- Apply the promotion gates and assign relabelling verdicts
- Select top candidates per type per arm from the discovery partition, stratified background only
- Emit core statistical results table

**Threshold stability**
- Recompute the passing set at every threshold value
- Compute Jaccard overlap against the primary threshold's passing set
- Emit threshold stability figure and the full threshold grid supplement

**Layer and architecture analysis**
- Compute the passing rate per layer per arm
- Compute mean enrichment among passers per layer per arm
- Identify the empirical peak layer per arm
- Compare the two 650M arms at the shared paired layer
- Compare the terminal layer across arms as the built-in negative control
- Compare the empirical peak against the intrinsic-dimension plateau
- Emit layer rate, layer strength, and architecture comparison figures

**Core statistical figures**
- Selectivity histogram against the chance reference line
- Cliff's delta histogram paired with the significance pass rate
- Occurrence versus magnitude scatter
- Dual-ranked bar charts by selectivity and by enrichment
- Activation frequency versus extent landscape

**Naive versus stratified comparison**
- Count passers under each background per type
- Count naive passers failing under stratification
- Count those triggering residue dominance
- Count those with chemically impossible support
- Compute fold-enrichment inflation and compare against the pre-stated prediction
- Emit the naive-versus-stratified table and figure

**Sequence-level track**
- Mean-pool activations per protein
- Rank-residualise against sequence length
- Run one-versus-rest Mann-Whitney per type with Cliff's delta
- Run the multi-class omnibus on singly-modified proteins as an optional secondary
- Emit the sequence-level supplement

**Targeted activation pass**
- Re-run inference at anchor layers for candidates plus random and activity-matched controls
- Store sparse activations at positive sites and sampled negatives, with residue positions and identities
- Store sparse full-profile vectors for a residue sample
- Compute protein-level event coherence against the control-matched universe

**Specificity validation**
- Cross-type enrichment vector within stratum, with top-to-second ratio
- Target-residue firing preference diagnostic
- Per-family precision-recall breadth, minimum and median
- Homology similarity check against a size-matched random protein set
- Three-tier negative robustness
- Crosstalk exclusion robustness
- Emit per-family breadth supplement

**Mechanistic grounding**
- Structural versus sequential activation clustering against a permutation null
- Sequence logo construction from top-activating positions
- Solvent accessibility correlation and secondary-structure composition
- Emit mechanism scatter, motif logo, structure renderings, accessibility supplements

**Representational integrity**
- Train and report the raw-activation ground-truth probe
- Run the feature absorption test
- Run the k-sparse probing sweep for feature splitting

**Generalisation and utility**
- Rebuild corpus, labels, and histograms for each out-of-distribution species
- Re-run the three-part decomposition on each species against species-specific nulls
- Recompute candidate statistics on the held-out partition and report retention ratios
- Recompute candidate statistics on the qPTM label set and report replication ratios
- Run the k-nearest-neighbour utility probe against length-only and composition-only baselines
- Emit cross-species figure and utility probe supplement

**Validation profile assembly**
- Merge every validation column into the per-candidate profile matrix
- Compute the pass-count distribution
- Emit validation profile table, heatmap, and pass-count histogram

**Ablation experiment**
- Decompose activations into reconstruction and error term
- Zero the target latent and splice the modified reconstruction back into the forward pass
- Measure KL divergence at PTM sites and at matched control positions
- Report the site-to-control ratio
- Compute the probe-based logit-difference variant for cross-paper comparability

**Steering experiment**
- Screen latents by encoder-decoder self-gain
- Solve the injection coefficient against true pre-activation
- Clamp the latent across the steering strength series
- Measure serine and threonine probability at the masked sequon position
- Measure proline probability at the intervening position
- Measure the position-wise probability profile around the site
- Measure collateral activation shift in non-target latents
- Record top-k membership changes for the TopK arm

**In-silico mutagenesis**
- Run loss-of-function sequon destruction
- Run gain-of-function sequon creation
- Run the specificity control mutation series
- Run the residue-substitution variant for non-motif types, reported as a differential against matched controls

**Causal baselines and assembly**
- Compute z-scores against random control latents
- Compute z-scores against activity-matched control latents
- Compute z-scores against non-SAE directions
- Emit causal figure panels
- Populate the four-cell interpretation matrix

**Ensemble analysis and release**
- Cluster candidate decoder directions per type
- Analyse within-cluster co-firing structure
- Run joint k-sparse probing against the best single latent
- Emit study design schematic and decoder embedding supplement
- Package code, processed datasets, histograms, full results, and null distributions

---

# PART 0b — DEGENERATE CASES AND STATISTICAL VALIDITY (CF-16…CF-24)

Found by a final sequential pass applying three lenses not used in earlier audits: **degenerate-case
behaviour** (what happens at n=0, n=1, all-zero, zero-variance), **statistical-validity consistency**
(are the tests mutually comparable), and **applicability** (does each check make sense for every
stratum, type, and species it will run on). These are lower-severity than CF-10/CF-13 — none is a
conceptual error — but each will either crash, silently emit NaN, or produce an uncomparable number.

### CF-16 — The discovery/held-out split is a **multi-label** stratification problem
`holdout_stratify_by: "ptm_type"` cannot be satisfied by a standard stratified split. Clusters are the
unit, a cluster contains proteins carrying **several** PTM types, and the types overlap — so there is no
single class label to stratify on. A naive `train_test_split(stratify=...)` will either error or silently
stratify on only one type. **Use iterative (multi-label) stratification**: assign clusters greedily,
at each step to whichever partition is furthest below its target for the type currently most
under-represented. Report achieved per-type proportions in both partitions; do not assume they are exact.

### CF-17 — Masked residues change the stratum population **per type**, not globally
Ambiguity-masked residues are excluded from both positives and negatives *for that type*. In histogram
terms they must be excluded from the tally entirely — **if the implementation only omits them from the
positive set, they fall into the negative set by default**, which is precisely the false-negative
injection the mask exists to prevent. Consequence: the stratum size $g_\ell$ is **per (stratum, type)**,
not one number per stratum, because different types mask different residues. Verify the enrichment
denominator uses the type-specific population.

### CF-18 — Dead latents divide by zero at normalisation
A latent that never fires during calibration has max activation 0, and the reciprocal rescale is
undefined. **Set its scale to 1.0, mark it dead, and exclude it from all downstream tallies** — do not
let a NaN or inf propagate into the histogram, where it will silently corrupt an entire row.

### CF-19 — **A direct consequence of CF-10**: latents with zero firings on modified residues
CF-10 restricts the label domain to PTM types only. A latent that fires exclusively on unmodified
residues therefore has an **all-zero count vector**, making selectivity (0/0), enrichment, and effective
types claimed all undefined. **This will be common, not rare** — most latents are not PTM-related.
Handle explicitly: these are not failures, they are latents with no PTM signal. Report them as a counted
category (`no_modified_firings`), exclude them from the three effect sizes and from the selectivity and
effective-types histograms, and include the count in the attempted-versus-analysed reporting so the
denominator of every rate is unambiguous.

### CF-20 — One-sided Fisher and two-sided χ² are not comparable
The plan specifies Fisher's exact as **one-tailed for enrichment**, but χ² is inherently **two-sided**.
Mixing them means the significance bar differs by a factor of two depending on which test the
expected-count rule happened to select — biasing which cells pass, in a way correlated with sample size.
**Make the direction consistent**: use one-sided in both cases. For χ², halve the p-value when observed
exceeds expected and use $1 - p/2$ otherwise. State the choice; a reviewer will check it.

### CF-21 — Type-specificity is **inapplicable** in single-type strata
The G3 gate compares top-type to second-type enrichment. **A stratum with only one surviving PTM type has
no second type**, and the ratio is undefined. This is not hypothetical: the asparagine stratum plausibly
retains only N-glycosylation after the ≥200-site filter, and **N-glycosylation is a primary target**.
Report G3 as **`not_applicable`** in that case — never as a failure, and never as an automatic pass.
Where a stratum has ≥2 types, the gate applies as written. State per-stratum applicability in the results
table so a reader can see which candidates the gate could and could not judge.

### CF-22 — Out-of-distribution species do not all support every PTM type
*E. coli* largely lacks N-glycosylation; several lysine acylations are far better characterised in
bacteria than in yeast. Running every type against every species produces meaningless nulls and
uninterpretable failures. **Build a species × type applicability matrix from the source databases before
the OOD run**, and only test type-species pairs with ≥200 sites in that species. Absence of a type in a
species is a fact about biology, not a failure of the latent.

### CF-23 — The empirical peak layer must be chosen on the discovery partition
The peak layer feeds the anchor set, which determines what enters the targeted pass and every downstream
check. Selecting it on the full corpus and then reporting results at that layer is the same selection
circularity CF-2 removes for candidates. **Select the peak on discovery**; the full-corpus layer curve
remains the reported figure.

### CF-24 — The permutation null supplies thresholds and FDR, **not** per-latent p-values
With 1,000 draws the smallest attainable permutation p-value is 0.001, while the Bonferroni bar sits near
$5\times10^{-7}$. **A permutation p-value can never clear Bonferroni**, so the two must not be conflated.
Division of labour, to be stated explicitly in Methods: **individual p-values come from χ²/Fisher**
(analytic, unbounded below); the **permutation null supplies percentile thresholds for effect sizes and
the FDR estimate**. Every "above the 95th null percentile" criterion is an effect-size threshold, not a
significance test.

---

# PART 0 — DESIGN INVARIANTS

Everything in Part 0 must hold across all modules. It is written first because modules reference it, not because it is executed first.

## 0.1 The three arms

| | **Arm A (primary)** | **Arm B (architecture control)** | **Arm C (scale control)** |
|---|---|---|---|
| Repository | `liambai/InterProt-ESM2-SAEs` | `Elana/InterPLM-esm2-650m` | `Elana/InterPLM-esm2-8m` |
| Base model | `esm2_t33_650M_UR50D` | `esm2_t33_650M_UR50D` | `esm2_t6_8M_UR50D` |
| Input dim | 1,280 | 1,280 | 320 |
| Architecture | **TopK** (Gao et al.) | **L1 (ReLU)** | **L1 (ReLU)** |
| Dictionary | 4,096 (3.2×) | 10,240 (8×) | 10,240 (32×) |
| Layers released | 4, 8, 12, 16, 20, **24**, 28, 32, **33** | 1, 9, 18, **24**, 30, **33** | 1, 2, 3, 4, 5, 6 |
| Training corpus | 1M UniProt | 5M UniRef50 | 5M UniRef50 |
| Activation scale | raw | normalized [0,1] shipped; **use `ae_unnormalized.pt`** | same as B |
| Licence | Apache-2.0 | MIT | MIT |
| Size/layer | 42 MB | ~100 MB | ~26 MB |

**Total: 21 layer-checkpoint combinations.**

**Note**: the HuggingFace organisation is `Elana/`, not `ElanaPearl/` (the latter is the GitHub handle). Only InterProt's 4,096 dictionary is publicly released despite the paper sweeping 2,048–16,384.

**Three controlled contrasts this enables** (none exists in the literature for any biological concept):

| Contrast | Varies | Held constant | Question |
|---|---|---|---|
| A vs B @ layer 24 | TopK vs L1 | model, scale, layer, tokenisation | Does architecture change which PTM latents emerge? |
| B vs C | 650M vs 8M | architecture, dictionary width | Does scale change PTM representation? |
| A vs B @ layer 33 | architecture | terminal layer | Does late-layer decline replicate across architectures? |

**Rejected checkpoints, with reasons** (recorded so they are not revisited):

| Candidate | Reason |
|---|---|
| `biohub/ESMC-6B-sae-layer60-k64-codebook16384` | Different model lineage (ESMC ≠ ESM-2); its 698 PTM features are GPT-5-generated. Using them would make this an audit of LLM labels rather than a test of unsupervised emergence. Retained as a **discussion comparator**. |
| `onkarsg10/Rep_SAEs_PLMs` (Gujral) | ESM2-35M; residue-level arm never statistically validated (LLM captioning only). Optional 4th arm if time permits. |
| `evillegasgarcia/sae_esm2_6_l3` | Trained on SCOPe (structural diversity, explicitly not annotation coverage); 573 dead latents. Retained as the **specific comparator** for the glycosylation novelty claim, since it is the only prior checkpoint reporting a glycosylation result. |

## 0.2 Configuration constants

An agent should place these in a single `config.py` / `config.yaml`. No constant may be hard-coded elsewhere.

```yaml
# Corpus
max_protein_length: 1022        # ESM-2 650M hard ceiling; both SAE families trained at this cap
organism_primary: "human"
organism_ood: ["mouse", "rat", "S_cerevisiae", "E_coli_K12"]
swissprot_reviewed_only: true

# Histogram / binning
n_bins: 64
bin_0_reserved_for_exact_zero: true   # mandatory: tau=0 test needs exact zero/non-zero split
bin_spacing: "log"                     # over (0.001, 1.0]; activation mass concentrates near zero
                                       # uniform spacing cannot represent tau=0.01

# Threshold sweep (values must land on bin edges)
tau_primary: 0.0
tau_sweep: [0.0, 0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 0.60, 0.80]

# Pre-filters
min_firings_residue_level: 50          # below this, contingency table is meaningless
                                       # and the latent inflates Bonferroni m for nothing
min_seq_mean_activation: null          # DISABLED - see CF-12. SPIRAL's 0.05 is calibrated to their
                                       # L0 ~12.6% density; Arm A (TopK k=64/4096) is ~8x sparser and
                                       # would be filtered to nothing. Coverage clause only.
min_sequences_seq_level: 10

# Label filters
min_sites_per_type: 200                # floor for own test; below -> pooled to "other PTM"
min_sites_headline: 1000               # bar for a headline claim (power analysis, §0.5)
lp_class_positive: ["I"]               # CPLM LP > 0.75
lp_class_masked: ["II", "III"]         # 0.25 <= LP <= 0.75 -> ambiguity mask

# Redundancy
cdhit_identity: 0.50                   # ProtSyntax threshold for PTM window data
cdhit_cluster_before_split: true

# Discovery / held-out split (cluster-level, assigned in M2)
holdout_fraction: 0.20                 # 20% of CD-HIT clusters reserved
holdout_seed: 42                       # fixed for reproducibility
holdout_stratify_by: "ptm_type"        # each type proportionally represented in both partitions
holdout_min_sites_per_type: 200        # a type falling below this in EITHER partition
                                       # is excluded from V12b, not from the main analysis

# Statistics
expected_count_min: 5                  # chi-square validity; below -> Fisher's exact
n_permutations: 1000                   # resolution 0.001 on permutation p
correction_primary: "bonferroni"
correction_secondary: ["benjamini_hochberg", "permutation_fdr"]
alpha: 0.05

# Candidate selection
top_n_candidates_per_type_per_arm: 20
anchor_layers: {A: [24, 33], B: [24, 33], C: [4, 6]}   # + empirical peak layer per arm
# Arm C anchors are depth-matched: layer 24 of 33 = relative depth 0.73; 0.73 x 6 = layer 4.
# Layer 4 is also VF's structurally strongest layer in ESM-2-8M. Layer 6 is terminal (matches 33).

# Validation gates
residue_dominance_fail: 7              # >=7 of top-10 positions same AA -> hard exclude
v1_type_specificity_min: 1.5           # top-type / second-type enrichment ratio
v6_crosstalk_retention_min: 0.50       # fraction of enrichment retained when crosstalk excluded

# Profile sample (V12c kNN probe) — must NOT borrow calibration_n_proteins
profile_sample_n_residues: 50000

# Causal
steering_multipliers: [0.0, 0.5, 1.0, 2.0, 4.0]
n_random_control_latents: 20
n_sequences_per_causal_experiment: 200
steering_min_self_gain: 0.05           # |alpha| = |decoder_row . encoder_row|; below this the
                                       # injection coefficient c = (target - preact)/alpha explodes
                                       # or flips sign -> latent excluded from steering, count reported

# Validation gate thresholds
v5_hard_tier_criterion: "significant_and_above_null"   # G4: NOT a borrowed retention ratio; see M16
```

## 0.3 The statistical test family — fixed, and why

SAE activations are **zero-inflated by construction**: ReLU produces exact zeros; TopK forces all but *k* of 4,096 latents to exact zero. ~99% of a typical latent's residue activations are exactly 0.0. This is a property of the SAE, not of the labels, so the test family is fixed independently of any dataset decision.

Three consequences:
1. **Cohen's *d* is degenerate.** VF's reported values (17.50, 15.83, 15.33) reflect *s*<sub>pooled</sub> → 0, not biology.
2. **Normality fails** — SPIRAL's stated reason for Kruskal-Wallis over ANOVA.
3. **A raw group difference conflates two distinct claims** — hence the decomposition below.

### The three-part decomposition

| Part | Question | Test | Effect size | What the effect size means |
|---|---|---|---|---|
| **1. Occurrence** | Does it *fire* disproportionately often on PTM sites? | χ² if all expected cells ≥ 5, **else Fisher's exact** | **Fold enrichment** $c_{i\ell^*}/\hat c_{i\ell^*}$ | Times more often than its own background rate predicts |
| | | | **Selectivity** $\max_\ell c_{i\ell}/\sum_\ell c_{i\ell}$ | Fraction of firings landing on the top type; chance = 1/L |
| | | | **Effective types claimed** $\exp\!\big(-\sum_\ell p_{i\ell}\log p_{i\ell}\big)$, where $p_{i\ell}=c_{i\ell}/\sum_{\ell'}c_{i\ell'}$ | The number of PTM types the latent *effectively* spreads across. 1.0 = perfectly monosemantic; 3.4 = behaves as if claiming ~3 types. **Severity, where selectivity gives only frequency.** |
| | | | **Odds ratio** $ad/bc$ | Companion to Fisher for rare types |
| **2. Magnitude \| firing** | *Given* it fires, does it fire *harder* on sites? | Mann-Whitney U, **restricted to residues where f > 0** | **Cliff's δ** | P(random site-activation > random non-site-activation), rescaled to [−1,1] |
| **3. Discrimination** | Does activation *rank* sites above non-sites? | **AUPRC** (primary), AUROC (secondary) | AUPRC itself | Chance baseline = the positive rate (~0.5–2%), not 0.5 |

> **⚠ CRITICAL SPECIFICATION — CF-10: the label domain $\ell$ for Eqs. 10–11 is PTM TYPES ONLY.**
> The formulae are inherited from SPIRAL, where **every** nucleotide carries one of 7 structure classes —
> there is no "none" class. **PTM labels are not like that**: within the lysine stratum, ~99% of lysines
> carry no modification at all. If "unmodified" is included in the label set $\ell$, then:
>
> - **Selectivity** $= \max_\ell c_{i\ell}/\sum_\ell c_{i\ell} \approx 0.99$ for **every latent**, because
>   almost all of any latent's firings land on unmodified residues. The metric becomes a constant.
> - **Enrichment** picks $\ell^* = \arg\max_\ell c_{i\ell} = $ "unmodified" for nearly every latent, giving
>   $\text{FE} \approx 1$ universally. The metric becomes meaningless.
> - **Effective types claimed** collapses to ≈1.0 for every latent, since the entropy is dominated by the
>   "none" mass.
>
> **This would silently destroy three of the pipeline's four occurrence effect sizes, with no error and no
> crash** — every latent would simply look identical and uninteresting, and the natural (wrong) conclusion
> would be "ESM-2 does not encode PTMs."
>
> **Specification**: for **selectivity (Eq. 10)**, **enrichment (Eq. 11)** and **effective types claimed**,
> $\ell$ ranges over **PTM types only**, and $c_{i\ell}$ counts the latent's firings on residues carrying
> type $\ell$. Firings on unmodified residues are excluded from these three numerators and denominators.
> Chance selectivity is therefore $1/L$ where $L$ = number of PTM types **in that stratum**, not $L+1$.
>
> **Where "unmodified" *is* included**: the one-vs-rest 2×2 occurrence test (its "label absent" cell is
> exactly the unmodified-plus-other-types population — unaffected by this, since a 2×2 has no label domain
> to choose), and the multi-class omnibus secondary on singly-modified residues, where it is a legitimate
> partition class.
>
> **Verification for the implementor**: if the notebook currently reports median selectivity above ~0.9
> across latents, or median enrichment near 1.0, this bug is present.

**χ² and Fisher are the same test** — both assess 2×2 association, differing only in whether the null is approximated or computed exactly. The choice is a per-cell numerical-validity rule, not a change of family.

**Separating occurrence from magnitude is novel.** No paper in the reading list does it. A latent firing on 80% of sites at moderate strength and one firing on 5% at saturation are mechanistically different and can produce identical *t*-statistics.

**Why "effective types claimed" is reported alongside selectivity** *(adapted from COMPASS-PTM)*. Selectivity is a single scalar reading only the maximum — it cannot distinguish a latent whose remaining firings are concentrated on one other type from one whose firings are smeared across eight. COMPASS-PTM makes exactly this distinction for multi-label predictors, separating the **frequency** of spurious extra labels (their conditional spurious rate: 0.654 for SAGEPhos vs 0.096 for COMPASS-PTM) from their **severity** (average extra labels per spurious residue: 1.30–1.76 vs 1.04–1.11). Exponentiated Shannon entropy is the natural latent-level analogue, is computed directly from the same count vector at zero cost, and gives a number on the same intuitive scale as theirs.

**Why AUPRC is primary — now with a PTM-specific published example.** COMPASS-PTM report that MusiteDeep achieves **0.964 AUROC on dbPTM-ML with a precision of only 0.179**, which they describe as rendering the predictions impractical for experimental follow-up. That is precisely the AUROC-inflation-under-rare-positives failure mode this pipeline avoids by making AUPRC primary, and it is a directly citable instance from the PTM literature rather than a general statistical argument.

**Framing terminology — the dual long-tail** *(COMPASS-PTM's term, adopted)*. PTM data carries two coupled imbalances: **(i) inter-class**, where a few common modifications hold most labelled instances (phosphorylation + acetylation + ubiquitination >90% of dbPTM sites) while many biologically important PTMs are rare; and **(ii) intra-class**, where positive sites are sparse against an overwhelming background of unmodified residues. This pipeline addresses (i) by per-type testing with per-stratum correction rather than pooled metrics, and (ii) by residue-stratified backgrounds plus AUPRC. Using the established term makes the design's motivation legible to reviewers of the PTM literature.

### Core formulae (SPIRAL Eqs. 9–11)

$$\chi_i^2 = \sum_\ell \frac{(c_{i\ell}-\hat c_{i\ell})^2}{\hat c_{i\ell}}, \qquad \hat c_{i\ell} = g_\ell\cdot\frac{\sum_{\ell'}c_{i\ell'}}{\sum_{\ell'}g_{\ell'}}$$

$$\text{selectivity}_i = \frac{\max_\ell c_{i\ell}}{\sum_\ell c_{i\ell}} \qquad \text{enrichment}_i = \frac{c_{i\ell^*}}{\hat c_{i\ell^*}},\ \ \ell^*=\arg\max_\ell c_{i\ell}$$

Cliff's δ, and its identity with Mann-Whitney U:
$$\delta = \frac{\#(A>B)-\#(A<B)}{n_A n_B} = \frac{2U}{n_A n_B}-1$$

### Tests considered and rejected

| Test | Source | Decision | Reason |
|---|---|---|---|
| Welch's *t* + Cohen's *d* | VF | **Secondary, footnoted** | Degenerate under zero-inflation; not scale-invariant across arms. Retained only for numerical comparability with VF Table 4. |
| AUROC as primary | VF | **Secondary** | At 0.5–2% positive rates it is misleadingly optimistic. Retained for VF's pass/mixed/fail band comparison. |
| Precision/recall at fixed τ = 0.80 | VG&A | **Rejected as primary** | No p-values, no null, no correction — the weakest bar in the literature and precisely what this work supersedes. Computed *only* for the head-to-head glycosylation claim in VG&A's own units. |
| Family-specificity F1 > 0.7 | InterProt | **Rejected** | Single classification metric, no significance, no correction. |
| Domain-adjusted F1 | InterPLM | **Rejected; contingency** | Solves granularity mismatch, not statistical rigour. PTM sites are single residues. Becomes correct **if** any retained annotation is region-level rather than site-level. |
| Mutual information | ESMC-SAE | **Rejected** | Single dependency score, no null, no correction; conflates occurrence and magnitude. |
| Graph-distance monosemanticity + LCA depth | Gujral | **Deferred** | Requires a PTM concept DAG — a project in itself. Named extension. |
| MCC, F_max, S_min, R², MAE | ProtSAE, ProtSyntax | **Rejected** | Downstream-classifier metrics; no classifier is the deliverable. |
| SCR / TPP | SAEBench | **Rejected** | Subsumed by the stratified design (§0.4), which removes the residue confound by construction rather than by ablation. |

## 0.4 The residue-stratified design — the central methodological decision

**The confound.** Every succinylation site is a lysine. Amino-acid-identity latents demonstrably exist (VF: L6/2417 fires 10/10 on alanine, L1/2023 10/10 on valine; InterPLM found hundreds even in SAEs trained on *randomised* ESM-2 weights).

With **all residues** as background, a pure lysine detector scores:
$$\text{FE} = \frac{P(\text{fires}\mid\text{succinylated})}{P(\text{fires})} = \frac{1.00}{0.058} \approx 17\times$$
— a 17-fold enrichment with an astronomically small p-value, for a latent that knows nothing about succinylation. It would top the results table.

With **lysines only** as background: FE = 1.00/1.00 = **1.00**. Exactly zero. The confound becomes *arithmetically impossible* rather than statistically corrected — stronger than rank-residualisation, where a residual always remains to argue about.

**Strata:**

| Stratum | ~% residues | ~Count | PTM types |
|---|---|---|---|
| **Lysine (K)** | 5.8% | ~375,000 | CPLM's 29 verified types — *Acylation (15)*: acetylation, **succinylation**, crotonylation, malonylation, 2-hydroxyisobutyrylation, β-hydroxybutyrylation, butyrylation, propionylation, glutarylation, lactylation, formylation, benzoylation, HMGylation, MGcylation, MGylation · *Ub/Ubl (4)*: ubiquitination, sumoylation, pupylation, neddylation · *Others (10)*: methylation, glycation, hydroxylation, phosphoglycerylation, carboxymethylation, lipoylation, carboxylation, dietylphosphorylation, biotinylation, carboxyethylation |
| **Ser/Thr** | 13.7% | ~890,000 | phosphorylation, **O-GlcNAcylation**, O-glycosylation |
| **Asparagine (N)** | 3.6% | ~235,000 | **N-glycosylation**, deamidation |
| **Cysteine (C)** | 2.3% | ~150,000 | S-nitrosylation, palmitoylation, glutathionylation, S-sulfhydration |
| **Arginine (R)** | 5.6% | ~365,000 | methylation, citrullination, ADP-ribosylation |
| **Tyrosine (Y)** | 2.7% | ~175,000 | phosphorylation, nitration, sulfation |

**Bonus**: this delivers VF's Stage 2 (cross-class) for free and in sharper form — within lysine, does a "succinylation latent" also fire on acetylation and ubiquitination? VF had to *construct* harder comparison classes; here they are given by chemistry.

**Both backgrounds are computed and reported.** The naive (all-residue) analysis is not a straw man — it is what InterPLM's `ft_mod_res`/`ft_carbohyd` F1, VG&A's glycosylation precision/recall, and ESMC-SAE's MI ranking all effectively do. Reporting the difference converts a methodological argument into a measured false-discovery count.

**Pre-stated prediction** (checked in M13): the inflation factor should approximate 1/*f*<sub>residue</sub> for pure identity detectors — ≈17× lysine, ≈28× asparagine, ≈7× Ser/Thr.

## 0.5 Reporting conventions — fixed before any result is seen

| # | Convention | Source | Reason |
|---|---|---|---|
| **R1** | Significance and effect size in **separate columns**, never merged | SPIRAL §6.2 | Mathematically independent. SPIRAL: 1,237/1,237 passed Bonferroni at median η² = 0.026. |
| **R2** | **Dual ranking** — top by selectivity *and* separately by enrichment | SPIRAL §5.4 | Only 2 of 10 overlapped in SPIRAL's RNA analysis. |
| **R3** | **Number-vs-strength** — % passing significance *and* mean enrichment among passers, as separate columns | SPIRAL Table 2 | They dissociate (L0: 97.5% at 1.04×; L5: 44.3% at 1.61×). Primary mitigation for the dictionary-width confound. |
| **R4** | **Attempted vs analysed** — dictionary size, alive, activity-filtered, tested | Gujral (819/20,000 vs 202/480) | Makes selection asymmetry between arms explicit. |
| **R5** | **Honest negative reporting** — nulls in main text | SPIRAL §8 (reported silhouette −0.197, ARI 0.085) | Establishes positives were not selected for. |
| **R6** | Every threshold carries its **null percentile**; VF bands as comparison column only | M8 | Makes clear which number drove which decision. |
| **R7** | **Value above bar** (absolute), **fold-improvement inside bar** (relative) | ESMC-SAE Fig 1 | Two numbers in one bar's space. |

## 0.6 Claim ladder — what may be asserted at what evidence level

| Level | Claim | Requires | H&F role |
|---|---|---|---|
| **L1** | "Latent X's activation is statistically associated with PTM type Y in checkpoint Z at layer L" | G1 + G2 | Evaluator |
| **L2** | "…specific to Y, not to the residue stratum or a protein family" | L1 + V1, V3, V4 | Evaluator |
| **L3** | "…not an artefact of annotation coverage or crosstalk leakage" | L2 + V5, V6 | Evaluator |
| **L4** | "…activation pattern consistent with Y's known biology" | L3 + V7, V8, V9 | Evaluator |
| **L5** | "…and ESM-2 uses this latent in its own computation at Y sites" | L4 + C1 (and C2/C3 where applicable) | Evaluator, causally grounded |
| **L6** | "ESM-2 encodes Y as a recoverable direction in its residual stream" | L5 replicated across arms | Evaluator, model-level |
| **L7** | "Latent X predicts a modification site absent from current databases" | L5 + independent cross-database confirmation | **Multitasker** |

**Never claimed**: that a latent corresponds to a causal mechanism *in protein biology* (causal work establishes only load-bearing within ESM-2's computation); that a specific latent index is canonical (single-checkpoint scoping).

**Teacher-role scoping**: validation is grounded in dbPTM/CPLM — databases built from accumulated wet-lab curation. State this precisely as **existing curated wet-lab data, not new wet-lab data.**

## 0.7 Acknowledged residual confounds

| Confound | Why it matters | Mitigation |
|---|---|---|
| **Dictionary width** (4,096 vs 10,240 in A-vs-B) | InterProt Fig 7b: expansion factor increases family-specific feature count. Arm B will mechanically yield more significant latents. Subsampling B is invalid — a random 4,096-subset of a 10,240 dictionary is not the dictionary an SAE would have learned at width 4,096; feature splitting means discarding latents destroys concepts. | R3 (rates not counts); Bonferroni per dictionary; percentile-against-own-null (M8) |
| **Training corpus** (1M UniProt vs 5M UniRef50) | Kissane et al.: SAE features are dataset-dependent. Cannot be eliminated without retraining. | **Bounded by the B-vs-C contrast** — B and C share corpus and architecture, differing only in scale. Close B/C agreement ⇒ corpus-driven variance within the L1 family is small. Divergence ⇒ the A-vs-B architecture attribution weakens, reported as such. |
| **Sequence leakage** (SAEs trained on UniRef50/UniProt ⊃ Swiss-Prot) | Looks fatal to a reader who has not thought it through. | **Non-issue, stated proactively**: the SAEs are unsupervised reconstruction models that never saw PTM labels. Label leakage is structurally impossible. Only a *predictive generalisation* claim would be compromised, and none is made. |
| **Seed instability** | P&B: SAEs differing only in seed share ~30% of latents; TopK less stable than L1. | **Acknowledged, not tested.** A cross-arm decoder-alignment check was considered and rejected: A and B differ in architecture, width, *and* corpus simultaneously, so non-alignment would be uninterpretable. A test whose negative result carries no information is not worth the compute. Scoping: the claim is that *a* latent encoding this PTM exists in *this* SAE, not that a specific index is canonical. |

## 0.8 Contingencies — pre-registered

R5 applies: these are main-text results, not failures.

| If | Then | Reported as |
|---|---|---|
| An arm trips the broken-instrument floor | Exclude that arm; report the figures | A reproducibility finding about published checkpoints |
| An arm has poor-but-functioning fidelity | **Retain**; report figures adjacent to every result from that arm | Lets the reader weigh the result against the instrument, rather than the pipeline deciding for them |
| No latent clears G1 for a type | Report the null **with its power analysis** | Bounded negative: "ESM-2 does not encode Y above enrichment E, detectable with n sites at layer L." E comes from the permutation null (M8). |
| All candidates fail G3 (stratum-general) | Report as the **primary finding** | Substantive: ESM-2 represents "modifiable lysine context" but not modification *type* — a claim about representational granularity, directly relevant to whether PTM-type prediction from PLM embeddings should work |
| Enrichment only against Background negatives | Report with annotation-bias caveat foregrounded | Quantifies annotation-coverage bias — what the three-tier design was built to measure |
| Enrichment collapses without crosstalk residues | Reattribute the latent to the co-occurring type | Demonstrates co-occurrence leakage empirically — no prior paper shows it |
| All causal tests null | Report VF's pattern replicating in a new domain | "Statistically strong latents are not causally load-bearing," corroborated on a different task, model, and modality |
| Arms A and B disagree substantially | Architecture finding; weaken the corpus-confound argument accordingly | TopK-vs-L1 divergence; bounds transferability of any single-checkpoint SAE result |
| Naive and stratified backgrounds agree | Report; revisit whether identity latents exist at these layers | Would contradict VF and InterPLM on amino-acid latents — surprising, worth investigating |

---

# PART 1 — EXECUTION MODULES

## Phase I — Data (CPU only, no GPU required)

### M1. Acquisition

**Inputs**: none.

**Computation** — download and verify:

| Source | What to fetch | Notes |
|---|---|---|
| **UniProt/Swiss-Prot** | Reviewed human proteome, FASTA + feature fields (`ft_carbohyd`, `ft_mod_res`, `ft_lipid`, `ft_signal`, `ft_transit`) | Cross-reference set; the exact fields InterPLM tested |
| **dbPTM 2025** | Full experimentally-validated site set | **2025 only** — releases are cumulative and approximately nested (3.0: 208,521 experimental → 2019: ~908,000 → 2022: 2,235,000 → 2025: 2,243,000). Earlier releases add duplicates, not coverage. |
| **CPLM 4.0** | `Homo sapiens.zip` from Section 1 of `cplm.biocuckoo.cn/Download.php` | **Do not touch Section 2** (Annotations) — that is where all ~45 GB lives (102 auxiliary resources; human alone = 699,440,281 annotation entries). Species-wise download gives all 29 lysine types in one file with consistent identifiers, which matters for crosstalk matching. Last updated 10 Feb 2025. |
| **qPTM** | Human site table (`qptm.omicsbio.info/download.php`) | **Required (CF-26)** — the independent-curation replication label set. Must be acquired before the streaming pass, not after |
| **O-GlcNAcAtlas 4.0** | Dataset-I (unambiguous) and Dataset-II (ambiguous) | Both needed — II drives the mask |
| **N-GlycositeAtlas** | Human N-glycosite table | Use the **14,000 sites** figure, not the 30,000 peptides |
| **N-GlyDE** | 30%-identity set (629 proteins, 1,547 glycosylated + 828 non-glycosylated sequons) and 60% set | **The 828 verified non-glycosylated sequons are the only Gold-tier negatives in the entire project** |
| **Models** | `facebook/esm2_t33_650M_UR50D`, `facebook/esm2_t6_8M_UR50D` | ~3 GB |
| **SAEs** | 21 checkpoints per §0.1; **`ae_unnormalized.pt` for Arms B and C** | ~1.2 GB |

**Unit definitions** — the databases count different entities; conflating them produces wrong numbers:

| Unit | Meaning |
|---|---|
| **Site** | One residue, one protein, one modification type |
| **Event** | One (protein, residue, type) triple. A residue with two types = 2 events, 1 residue |
| **Unique modified residue** | The residue, counted once regardless of type count |
| **Sequon** | A candidate motif (N-X-S/T for N-glycosylation), modified or not |
| **Glycosite-containing peptide** | An MS-identified peptide; many map to one site |
| **Unambiguous site** | MS localised the modification to one specific residue |
| **Ambiguous site** | Peptide known modified; spectrum could not distinguish which candidate residue |

**Verified counts:**

| Database | Figure | Unit |
|---|---|---|
| dbPTM 2025 | 2.79M total; **2.243M experimentally validated** | sites |
| dbPTM 2019 (last public per-type breakdown) | phosphorylation 571,032; acetylation 137,442; ubiquitination 118,495 | verified sites |
| dbPTM 2022 | phosphorylation >63.9%; ubiquitination ~16.4% | proportion |
| Oct-2020 snapshot | top-3 ≈ 827,000 of ~908,000 (>90%); **24 types** have >80 sites; **Lys carries 15 PTM types**, Cys and Ser ≥10 | sites |
| CPLM 4.0 | **592,606 events** on **463,156 unique lysines** in **105,673 proteins**; 29 types, 219 species | events/residues/proteins |
| CPLM crosstalk | 10,746 shared sites; ≤53.5% of acetylation and ≤33.1% of ubiquitination co-occur; 76 co-occurrence types | sites |
| CPLM LP classes | **I (LP>0.75): 141,068 (99.25%)** · II: 659 (0.46%) · III: 405 (0.28%) · **IV: already removed by CPLM** | LP-scored subset (~142,132), not all events |
| PLMD 3.0 | 18,593 succinylation sites in 6,377 proteins | sites |
| O-GlcNAcAtlas 4.0 | **>19,000 unambiguous** + >11,000 ambiguous; >8,000 proteins; Ser:Thr ≈ 62:38 | sites |
| N-GlycositeAtlas | >30,000 peptides → **>14,000 sites** → >7,200 proteins | peptides→sites→proteins |
| N-GlyDE (30% id) | 629 proteins: **1,547 glycosylated + 828 non-glycosylated sequons** | sequons |
| N-GlyDE (60% id) | 832 proteins: 2,050 + 1,030 sequons | sequons |

**Derived, and important**: in N-GlyDE's 30% set, **65.1%** of sequons in known glycoproteins are glycosylated (1,547/2,375) — a sequon in a glycoprotein is more likely modified than not, so "unannotated sequon = negative" is wrong roughly two-thirds of the time in that population. This single statistic justifies the three-tier negative hierarchy.

**Also derived**: CPLM's event:residue ratio 592,606/463,156 = **1.28** ⇒ ~28% of modified lysines carry multiple types. That is the crosstalk rate, measured not estimated.

**Interpretation of the >90% figure** (a common confusion): dbPTM holds 130+ types, but phospho + acetyl + ubiquitin account for >90% of the *sites*. The remaining <10% is spread across 127+ types. **dbPTM contains only modified sites — there is no "unmodified" category**; unmodified residues come from the corpus (M2).

**Emits**: raw data in `/kaggle/temp`; a manifest with SHA sums and access dates.

**Cost**: minutes to ~1 h depending on bandwidth. No GPU.

**Practical note**: extract CPLM once, retain only (accession, position, type, LP, PMID, flanking peptide), write a single Parquet (<50 MB), and **upload as a Kaggle Dataset** so analysis notebooks attach a small static file instead of re-processing.

---

### M2. Corpus construction

**Inputs**: M1 Swiss-Prot.

**Computation**:
1. Filter to reviewed human, length ≤ 1,022 → expect **~17,000–18,000 proteins**.
2. Compute exact residue counts per stratum (K, S/T, N, C, R, Y). **These become $g_\ell$ and must be exact, not sampled** — every enrichment value depends on them.
3. Compute total residue count. *Estimate ~6.5M (≈17,500 × ≈375 mean length after the cap); the true value could plausibly fall in 5.5–7M and is computed here, not assumed.*
4. CD-HIT at 50% identity; assign cluster IDs. **Cluster before any split**; all sites from a cluster stay together.
5. **Assign each cluster to `discovery` (80%) or `holdout` (20%)**, seed 42, stratified so each PTM type's sites are proportionally represented in both partitions. Store as a `split` column on the protein table.

**Why the split exists and why it is at cluster level.** M9 selects candidate latents by taking the *maximum* enrichment across ~1,200 tested latents. The maximum of a noisy quantity is optimistically biased — reporting that same enrichment as the result is circular. V12b (M19) recomputes enrichment for the selected candidates on held-out clusters, giving an unbiased estimate; the discovery-minus-holdout gap quantifies the selection optimism directly.

The split must be at **cluster** level, not protein or site level, because homologous proteins share PTM sites and local sequence context. A protein-level split would place a protein in discovery and its 80%-identical homolog in holdout, leaking the signal across the boundary and making the "held-out" estimate meaningless.

**Feasibility check**: 20% of succinylation's expected ~2,000–5,000 corpus sites leaves ~1,600–4,000 for discovery — still above the ≥1,000 headline bar. Types falling below 200 sites in either partition are excluded from V12b only; they remain in the main analysis, which uses the full corpus.

**Why this corpus definition:**

| Constraint | Reason |
|---|---|
| ≤1,022 residues | ESM-2 650M's hard ceiling; both SAE families trained at this cap. Not a choice. |
| Swiss-Prot reviewed | Both arms' normalisation statistics originate here, so activation scales are in-distribution; matches InterPLM's own evaluation corpus (50k Swiss-Prot <1,024 aa) |
| Human | dbPTM's densest, best-curated species; maximises positives per unit of compute |
| **All** qualifying proteins, not a sample | Makes $g_\ell$ **exact rather than estimated** |

**Why not larger**: the binding constraint on power is positive-site count per type, not background size. More proteins add background without adding sites for rare types.

**What the corpus is for** — three roles, and the first is the one usually misunderstood:
1. **Enrichment is undefined without it.** $\hat c$ derives from $g_\ell$. With positives only, every rate is 1.0 and enrichment is identically 1.
2. **Latent firing rates need estimating.** Selectivity requires knowing how often each latent fires overall within the stratum.
3. **Precision of the denominator.** $P(\text{fires}\mid\text{any K})$ estimated from 375,000 lysines is precise; from 3,000 it would be noisy, and that noise propagates into every FE value.

Under stratification, the corpus's negatives-supplying role and its denominator role are the *same population* — the denominator runs over all lysines = succinylated ∪ non-succinylated.

**Emits**: `corpus.parquet` (accession, sequence, length, cluster_id, **split**); `stratum_counts.json`; `split_manifest.json` (cluster→partition assignment, seed, per-type site counts in each partition).

**Gate**: protein count within 15,000–20,000. Outside that range, re-check the Swiss-Prot release and length filter. Additionally, each of the three primary types (N-glycosylation, O-GlcNAcylation, succinylation) must retain ≥1,000 sites in the **discovery** partition; if not, reduce `holdout_fraction` to 0.15 and re-split rather than proceeding.

---

### M3. Label construction — the filtering cascade

**Inputs**: M1 databases, M2 corpus.

**Computation** — six steps, in order, with counts recorded after each.

**Step 1 — Evidence tier.** Parse each dbPTM record's evidence annotation. Retain MS/MS-derived and low-throughput experimental. Discard UniProt-derived "by similarity" / "potential" / "probable" (dbPTM 3.0 documented 226,122 such putative sites) and text-mining-only records lacking an experimental citation.
*Why*: adopted from ProtSyntax. Homology-transferred annotations are predictions; including them would test whether the SAE agrees with another algorithm rather than with experiment.

**Step 2 — Site-mapping validation.** For every (accession, position, type), retrieve the UniProt canonical sequence and assert the residue matches the expected chemistry (K for succinylation, N for N-glycosylation, S/T for O-GlcNAc…). Log and drop failures. **Report failure rate per source database.**
*Why*: dbPTM integrates 48 sources with differing sequence versions and indexing conventions (0- vs 1-based). Off-by-one and stale-isoform errors are real and silent.
*CPLM bonus*: reference sequences ship with the site data, so validate against CPLM's own reference **first**, isolating whether a mismatch originates in CPLM's indexing or a UniProt version difference. The **15-aa flanking peptide** gives a second free check — its centre must be a lysine and it must occur in the reference sequence at the stated position.

**Step 3 — Ambiguity masking.** Three source-specific implementations, one common treatment.

- *3a — O-GlcNAcAtlas (binary)*: for Dataset-II entries, identify the peptide and every candidate Ser/Thr within it; add all to the exclusion mask.
- *3b — CPLM (continuous, via LP)*: LP is MaxQuant's localisation probability — a quantitative measure of how confidently the modification was localised to *that* lysine. Class I → positive; **Classes II and III → masked**; Class IV already removed by CPLM. Sites *without* an LP score (low-throughput, literature-curated) are retained as positives — LP is only assigned by high-throughput pipelines, so its absence indicates a different evidence route, not low confidence. *Practical note*: II + III = 1,064 sites (0.74%), so this mask is small for CPLM — unlike O-GlcNAcAtlas, where ~37% of sites are ambiguous.
- *3c — dbPTM*: no documented per-site confidence field. Sites originating from CPLM/PLMD inherit LP treatment through deduplication (Step 4).

*Common treatment*: masked residues excluded from **both** positive and negative sets for that type.

*Why*: an ambiguous site cannot be a positive (the modification may be on a neighbour) and must not be a negative (the modification genuinely exists somewhere in that peptide, so labelling all candidates unmodified injects guaranteed false negatives).

> **CRITICAL IMPLEMENTATION NOTE.** Masking is a **statistics-layer** operation, never a sequence-layer one. The full intact sequence goes to ESM-2; every residue is tokenised and every activation computed, so context is preserved exactly. Masked positions are simply not tallied into any contingency-table cell. **Deleting residues from the sequence would shift downstream positions and corrupt activations at every neighbouring residue.**

**Step 4 — Cross-source deduplication.** Merge on (accession, position, type). Retain `source_count` = number of independent databases/publications reporting each site.
*Why*: dbPTM already integrates UniProtKB and PhosphoSitePlus; O-GlcNAcAtlas, N-GlycositeAtlas and CPLM overlap with it and each other. Without merging, a site reported by four sources is counted four times. `source_count ≥ 2` defines a **high-confidence subset** — a free robustness analysis.

**Step 5 — Minimum frequency.** Count surviving sites per type per stratum. Types with <200 sites are **relabelled into "other PTM"** for that stratum, not deleted. Types with ≥1,000 marked headline-eligible.
*Why pool rather than delete*: a rare-type site is still a modified residue; treating it as an unmodified negative would be factually wrong. The ≥200 floor comes from contingency-table feasibility; ≥1,000 from the power calculation (to detect FE = 2 at Bonferroni α with m ≈ 10⁵, i.e. α_adj ≈ 5×10⁻⁷).

**Step 6 — Redundancy.** CD-HIT clusters from M2 applied; all sites in a cluster assigned together, and each site inherits its cluster's `split` label (`discovery` / `holdout`).
*Threshold precedents*: **50%** (ProtSyntax, for PTM window data — adopted here as the closest analogue), **40%** (COMPASS-PTM, MMseqs2, protein-level split "to minimize redundancy and evolutionary leakage"), **30%** (InterProt, MMseqs2, Swiss-Prot family evaluation). Ours is the loosest of the three, chosen because ProtSyntax's task — residue-level PTM windows — matches this one most closely. **A 40% sensitivity re-run is cheap** (re-cluster, re-split, re-tally from stored histograms — no new inference) and worth reporting if any result looks homology-driven under V4.
*Note*: this controls **within-corpus** redundancy. It does not address overlap with the SAEs' training corpora — see §0.7, where that is argued to be a non-issue for the claims made.

**Then construct:**

*Three-tier negatives:*

| Tier | Definition | Available for |
|---|---|---|
| **Gold** | Experimentally verified unmodified | N-glycosylation only (N-GlyDE, ~828 sequons) |
| **Hard** | Unannotated target residue in a protein known to carry that PTM (Corpus A) | All types |
| **Background** | All other target residues (Corpus B) | All types |

*Two corpus definitions:*

| Corpus | Definition | A "negative" means |
|---|---|---|
| **A (primary)** | Restricted to proteins with ≥1 annotated site of the type under test | "Studied for this PTM, this residue unmodified" |
| **B (sensitivity)** | Proteome-wide | Contaminated by never-studied proteins |

*Why both*: dbPTM's coverage is biased toward well-studied proteins; a residue in an unstudied protein is not evidence of non-modification. The difference between A and B is itself a reportable quantity.

*Crosstalk flags*: mark every residue carrying >1 type; record the type combination.

*Two label assignments* — **stratified** (per §0.4) and **naive** (all residues as background). Both are carried forward through M6–M13.

**Emits** — exact schemas, so the "negative population" is unambiguous:

`labels_stratified.parquet` — **one row per (residue, PTM type) pair that is testable for that type**:

| Column | Meaning |
|---|---|
| `accession` | UniProt accession |
| `position` | 1-based residue index in the canonical sequence |
| `residue` | the amino-acid letter at that position |
| `stratum` | the residue-identity stratum (`K`, `ST`, `N`, `C`, `R`, `Y`) |
| `ptm_type` | the type this row is scored for |
| `label` | 1 = annotated site of this type, 0 = negative |
| `negative_tier` | `gold` / `hard` / `background`, null when `label = 1` |
| `corpus` | `A` (annotated-protein-restricted) or `B` (proteome-wide) |
| `split` | `discovery` / `holdout`, inherited from the cluster |
| `cluster_id` | CD-HIT cluster |
| `crosstalk` | 1 if this residue also carries another PTM type |
| `source_count` | number of independent sources reporting this site |
| `masked` | 1 if ambiguity-masked — **excluded from both positives and negatives** |

`labels_naive.parquet` — identical schema, except `stratum` is the placeholder `ALL` and the negative
population is every residue in the corpus rather than only residues of the target type.

`labels_qptm.parquet` — identical schema to the stratified file, labels drawn from qPTM (CF-26).

> **What "negative population" means.** For a given (latent, PTM type) test it is **the set of rows with
> `label = 0` that the test scores against** — i.e. the denominator of enrichment and the "label absent"
> column of the contingency table. It is defined by three fields together: `stratum` (which residues are
> even eligible), `negative_tier` (how much the absence of annotation is trusted), and `corpus` (whether
> unstudied proteins are included). Changing any one changes the population and therefore the number.
> Rows with `masked = 1` belong to **no** population — they are dropped from both sides.
- `exclusion_mask.parquet`
- **T1 — Dataset provenance**: rows = the six steps; columns = sites entering, removed, remaining, reason, per source
- **T2 — Label composition**: rows = type × stratum; columns = positive sites, **sites in discovery**, **sites in holdout**, stratum size, positive rate, crosstalk fraction, negative tiers available, eligibility flags (≥200 floor, ≥1,000 headline bar, V12b-eligible)
- **F2 — Dataset composition**: (a) filtering waterfall; (b) label frequency per type within stratum, **log scale** — phosphorylation exceeds succinylation by ~2 orders of magnitude and a linear axis would render the targets invisible; (c) crosstalk fraction per type; (d) three-tier negative counts for N-glycosylation; (e) protein-length distribution, positive vs negative

**Gate**: N-glycosylation, O-GlcNAcylation, and succinylation must each clear the ≥1,000 headline bar. Expected surviving counts: N-glycosylation ~8,000–12,000; O-GlcNAcylation ~7,000–11,000; succinylation ~2,000–5,000.

**Cost**: hours, CPU only.

---

## Phase II — Instrument (GPU)

### M4. Model loading, Pass 0 calibration, normalisation

**Inputs**: M1 models and SAEs, M2 corpus.

**Computation**:
1. Load ESM-2 650M and 8M in fp16.
2. Load all 21 SAE checkpoints. **Use `ae_unnormalized.pt` for Arms B and C.**
3. **Calibration scan** over ~2,000 Swiss-Prot proteins: record each latent's maximum activation.
4. Derive normalisation constants; rescale encoder and decoder weights reciprocally so activations fall in [0,1] while reconstruction is preserved exactly. **Apply the identical procedure to all three arms** (InterPLM's `normalize.py`).
5. Fix histogram bin edges: **bin 0 = exactly zero**; bins 1–63 span (0.001, 1.0], constructed by **first pinning the eight non-zero τ sweep values as exact edges** (0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 0.60, 0.80), then log-spacing the remaining edges between them to reach 64 bins total.
6. **Dead-latent census**: latents never firing in calibration.

*Why normalise ourselves rather than use B's shipped normalised weights*: two checkpoints normalised by different people against different calibration sets are not comparable; one script against one calibration set eliminates provenance mismatch rather than adjusting for it.

*Why bin 0 must be reserved*: the τ = 0 primary test requires an exact zero/non-zero split; any bin spanning [0, ε) would contaminate it.

*Why log spacing*: activation mass concentrates near zero. Uniform 64-bin edges fall at multiples of 0.0156 and cannot represent τ = 0.01. Log spacing does not distort rank statistics, which need only monotonic bin ordering.

*Why the τ values must be pinned rather than assumed*: pure log spacing over (0.001, 1.0] with 63 bins places edges at $10^{-3+3k/63}$. That set happens to include 0.01 (k = 21) but **not** 0.05, 0.15, 0.25, 0.60 or 0.80. A naive log-spaced implementation would therefore make most of the τ sweep unreachable and M10's threshold-stability analysis would silently interpolate. **Pin first, fill second.**

> **MANDATORY FIRST ACTION.** Before the full corpus, run a **200-protein timing pass and extrapolate**. Throughput estimates below are unmeasured and sensitive to the sequence-length distribution, because attention cost is quadratic.

**Emits**: `normalization_constants.json`; `bin_edges.npy`; `dead_latents.json`; timing extrapolation.

**Cost**: ~10 min GPU.

---

### M5. Fidelity gate + intrinsic dimension

**Inputs**: M4.

**Computation**:
1. Per arm per layer, on **this corpus**: explained variance, cosine similarity, KL divergence (original vs SAE-spliced model outputs), % Loss Recovered, L₀.
   $$\%\text{Loss Recovered} = \left(1-\frac{CE_{\text{recon}}-CE_{\text{orig}}}{CE_{\text{zero}}-CE_{\text{orig}}}\right)\times100$$
2. **TwoNN intrinsic dimension** across all 33 ESM-2-650M layers. For each point, the ratio μ = r₂/r₁ of distances to first and second nearest neighbours; its distribution across points depends only on manifold dimension. Plot ID vs depth; mark the plateau.
3. Attempted-vs-analysed counts (R4): nominal dictionary, alive latents, latents passing the ≥50-firing filter.

*Why re-measure published fidelity*: a published number was measured on the publisher's corpus. If Arm A's EV on human Swiss-Prot is materially below its published value, every downstream claim from that arm is suspect — better discovered in ten minutes than in the discussion section.

> **⚠ CORRECTION — CF-25: report fidelity, do not gate on borrowed benchmarks.**
> The earlier specification said an arm "failing fidelity" is excluded from all claims, with reference
> numbers from SPIRAL (EV > 0.99997) and VG&A (L₀ ≈ 18). **Those numbers are not applicable.** SPIRAL's
> figure is from an SAE on an RNA language model with a different architecture, corpus and dictionary
> width; VG&A's L₀ is from a 10× SAE on ESM-2-8M trained on SCOPe. Excluding a checkpoint from a
> Nature-targeted study because it misses a threshold borrowed from an unrelated model would be
> indefensible.
>
> **What published fidelity numbers actually exist for the three arms:**
>
> | Arm | Published fidelity | Source |
> |---|---|---|
> | InterPLM-8M (scale control) | **% Loss Recovered per layer: 99.61, 99.32, 99.02, 98.40, 99.32, 100.00** | InterPLM preprint, Table 1 |
> | InterPLM-650M (architecture control) | **none published** | later release, no fidelity table |
> | InterProt-650M (primary) | **none published** | paper reports training loss only |
>
> So a comparative gate is available for exactly one of three arms, and it is the least important one.
>
> **Corrected specification, two layers:**
>
> **(a) Broken-instrument floor — the only hard gate.** An arm is excluded only if the SAE is
> *demonstrably not functioning*: explained variance < 0.90, **or** % Loss Recovered < 50%, **or** the
> reconstruction performs worse than zero-ablating the layer (Loss Recovered ≤ 0), **or** more than 50%
> of latents are dead. These are "something is fundamentally wrong" levels, not quality bars, and no
> healthy checkpoint approaches them.
>
> **(b) Mandatory reporting, no threshold.** Every metric is reported per arm per layer in the instrument
> table, alongside the published value where one exists and "not published" where it does not. Where a
> published value exists (InterPLM-8M) the comparison is stated. Where none exists, the number stands on
> its own and **the reader decides**. This is the honest position: the fidelity of these two 650M
> checkpoints on a human Swiss-Prot corpus is a **new measurement**, and reporting it is a contribution
> rather than a hurdle.
>
> **Consequence for interpretation**: an arm with poor-but-functioning fidelity is not discarded. Its
> results are reported with the fidelity figures adjacent, so a weak result at low explained variance is
> visibly confounded and a strong result at low explained variance is visibly surprising. Both are more
> informative than exclusion.

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-1: report the normalisation overflow rate.**
> Normalisation constants are fit on a ~2,000-protein calibration sample (M4), so corpus activations
> can and will exceed 1.0. The implementation correctly clamps these into the top bin rather than
> overflowing — but **the frequency of clamping is a diagnostic that is currently not reported.**
> If a meaningful fraction of a latent's firings exceed 1.0, the top bin becomes a heavy-tailed dump
> and every rank statistic derived from it (Cliff's δ, AUPRC, AUROC) loses resolution at exactly the
> end of the distribution that matters most.
>
> **Add to T3, per arm per layer**: the fraction of non-zero activations that clamped, and the count
> of latents whose clamp rate exceeds 1%. **Action rule**: if the corpus-wide clamp rate exceeds 1%,
> raise `calibration_n_proteins` and re-run M4 before proceeding — the calibration sample was too
> small to characterise the activation range.

**Emits**:
- **T3 — Instrument validation**: rows = arm × layer; columns = EV, cosine, KL, %LR, L₀, nominal dict, alive, tested
- **F3**: (a) fidelity metrics vs layer, three arms; (b) % Loss Recovered vs layer; (c) TwoNN ID vs all 33 layers with plateau marked; (d) alive/dead/tested counts

**Gate — instrument.** An arm is excluded **only** if it trips the broken-instrument floor above (EV < 0.90, or % Loss Recovered < 50%, or ≤ 0, or > 50% dead latents). Otherwise it proceeds, with all fidelity figures reported and carried alongside its results.

**Cost**: ~30 min GPU.

*F3c carries a prediction to check against M11: does the ID plateau coincide with the PTM-enrichment peak? If so, that cross-validates VG&A's heuristic on a task it was never designed for.*

---

## Phase III — Measurement (GPU)

### M6. Pass 1 — histogram accumulation

**Inputs**: M2 corpus, M3 labels (both assignments), M4 constants.

**Computation** — one streaming pass over the corpus:
1. ESM forward with `output_hidden_states=True` (returns all 34 hidden states in one pass).
2. For each of the 21 (checkpoint, layer) combinations: encode through the SAE (one matmul, negligible beside the transformer forward).
3. Bin each activation; `scatter_add` into per-(latent, label) histograms — **for both the stratified and naive label assignments, and separately for the `discovery` and `holdout` partitions**.
4. Additionally accumulate **exact running sum, sum-of-squares, and firing counts** per (latent, label), per partition.
5. **Accumulate a (latent × 20 amino acids) firing-count matrix**, and maintain a running top-*k* heap (*k* = 20) of highest-activating (position, residue identity) pairs per latent.
6. Accumulate **per-protein mean activation** per latent (feeds M14).
7. **Discard activations every batch.**

> **WHY STEP 5 EXISTS — a dependency fix.** Gate G2 (residue-identity dominance: ≥7 of the top 10 activating positions sharing one amino acid) is applied during candidate selection in **M9**, but per-residue identity is otherwise only available from Pass 2 (**M15**), which runs *after* M9. That is circular. Accumulating the amino-acid firing matrix and a top-*k* heap during Pass 1 breaks the cycle: G2 becomes computable at M9 from Pass 1 outputs alone. The cost is a 4,096 × 20 integer matrix plus a 20-element heap per latent — kilobytes. M16's V2 then re-confirms the same criterion on the richer Pass 2 data, so the check is applied twice with the second pass as verification, not as first computation.

*Why partition-separated histograms*: this is what makes V12b (M19) free. Discovery histograms drive candidate selection in M9; holdout histograms give the unbiased re-estimate — with no second inference run. Full-corpus statistics for the main analysis are recovered by simply summing the two partitions' histograms, since counts are additive.

> **WHY ACTIVATIONS ARE NEVER CACHED.** 13 unique layers × 1,280 dims × 2 bytes = 33 KB/residue. At ~6.5M residues that is **233 GB**, against Kaggle's 20 GB `/kaggle/working`. The entire pipeline design follows from this constraint.

**What histograms buy** — a histogram $H_{i\ell}[b]$ counts residues carrying label ℓ where latent *i*'s normalised activation fell in bin *b*:

- **Counts at any τ** = suffix sum $\sum_{b>b(\tau)} H_{i\ell}[b]$. The entire nine-value τ sweep is **nine array slices**, not nine passes. This is the single largest saving in the design.
- **Full 2×2 tables at any τ** — all four cells from suffix sums of two histograms. Feeds χ²/Fisher, FE, selectivity, odds ratio.
- **Means and variances exactly** — from the separately accumulated Σf and Σf², not bin centres. Matters because Welch's *t* and Cohen's *d* are reported for VF comparability and should not carry discretisation error.
- **Rank statistics to bin resolution** — Mann-Whitney U, AUROC, AUPRC computable by iterating bin pairs with mid-rank tie correction. Exact values recomputed in M15 for candidates, so the approximation only affects *which latents get promoted*, never a reported headline number.
- **What they cannot do**: they discard *which residue* produced each activation. Anything needing per-residue identity — distance correlations, residue-dominance, sequence logos, co-firing — requires M15.

**Parallelism**: shard the corpus in half across the two T4s as **two independent processes**, not `nn.DataParallel` (inefficient for this access pattern). Roughly halves wall-clock.

**Emits**: `histograms/{arm}_{layer}_{stratified|naive}_{discovery|holdout}.npy` (~2 GB total — four variants per combination, still trivial); `moments/*.npy`; `aa_firing_matrix/*.npy` (latent × 20); `top_activating/*.json` (top-20 position/residue pairs per latent); `protein_means/*.npy`.

**ADOPTED — cross-database replication label set (CF-26).** A **third** label assignment, derived from
**qPTM**, is tallied in the same pass. It costs only additional `scatter_add` targets over activations
already computed, so the marginal inference cost is zero — but **it must be built before this pass runs**,
because adding it later requires re-running all inference.

**What it tests, and why it is distinct from everything else in the pipeline.** The three-tier negatives
test annotation **coverage** — is a residue unmodified, or merely unstudied? Cross-database replication
tests annotation **provenance** — would a latent that looks enriched under dbPTM's curation still look
enriched under an independently curated label set on the *same corpus, same proteins, same residues*?
Nothing else in the pipeline asks this. It is the analogue of COMPASS-PTM's zero-shot evaluation of a
dbPTM-trained model on the independently built PTMint benchmark.

**Why qPTM rather than PTMint**: qPTM covers human, mouse, rat and yeast with site-level resolution across
multiple PTM types, so it overlaps the strata directly. PTMint is scoped to PTM regulation of
protein–protein interactions and is much smaller, making it a poor replication set for most types.

**How it is reported**: for each candidate latent selected on the dbPTM labels, recompute the three-part
decomposition on the qPTM labels and report the **replication ratio** = qPTM enrichment ÷ dbPTM enrichment,
alongside whether it remains significant. A ratio near 1 means the association is a property of the
biology; a ratio far below 1 means it is partly a property of dbPTM's particular curation. Types absent
from qPTM are reported `not_applicable`, never as failures.

**Histogram cost**: this adds a third label assignment, so the histogram store grows from four variants per
combination (stratified/naive × discovery/holdout) to six (adding qPTM-stratified × discovery/holdout).
Roughly 3 GB total — still trivial.

**Gate**: histogram row sums, summed across partitions, must equal known label counts from T2. A mismatch means a masking, indexing, or split-assignment bug.

**Cost**: ~1–2 h GPU for the 650M arms; ~5 min for the 8M arm. **Budget**: ~3 GB VRAM against 16 GB/T4.

---

### M7. Statistical testing — residue level

**Inputs**: M6 histograms and moments.

**Computation**, per (arm, layer, stratum, type, background, τ):
1. Build the 2×2 contingency table at τ.
2. Compute expected cell counts $E = (\text{row}\times\text{col})/n$; select **χ² if all E ≥ 5, else Fisher's exact** (one-tailed for enrichment).
3. Occurrence effect sizes: fold enrichment, selectivity, **effective types claimed** (exponentiated entropy of the label-count vector, §0.3), odds ratio.
4. Magnitude: Mann-Whitney U and Cliff's δ, **restricted to residues where f > 0**.
5. Discrimination: AUPRC (primary) and AUROC (secondary).
6. Secondary: Welch's *t*, Cohen's *d* (flagged degenerate).
7. **Chemically-impossible-association flag** (naive background only): for each significant (latent, type) pair, record the fraction of the latent's firings on that type's positives that occur on residues **chemically incapable** of carrying it — e.g. a "succinylation" association supported by firings on non-lysine residues. Under the stratified background this fraction is 0 by construction; under the naive background it is not, and it becomes a second, chemistry-grounded false-positive counter in M13.
8. Apply the activity pre-filter and record attempted vs analysed (R4). **Filter scope**: ≥50 firings *within the stratum* for the stratified background; ≥50 firings *corpus-wide* for the naive background, since the naive analysis has no stratum.

*Arithmetic for the χ²/Fisher decision*: a latent firing on 1% of ~375,000 lysines gives ~3,750 activations; a type at 1% of lysines gives E ≈ 37 → χ² valid. A 200-site type with a sparse latent (0.1% firing) gives E ≈ 0.2 → Fisher required. **Both will be needed; the rule decides, not a global choice.**

**Emits**: `results_raw.parquet` — the substrate of T4.

> **⚠ IMPLEMENTATION-GAP CHECK — CF-3.** Three M7 outputs postdate the plan revision the notebook was
> built against and do not appear anywhere in the changelog. Verify each exists before proceeding:
> **(a)** `effective_types_claimed` — exponentiated Shannon entropy of the label-count vector (§0.3);
> **(b)** the **chemically-impossible-association fraction** (step 7), computed for the naive background
> only and consumed by T5/F4; **(c)** the naive-background Bonferroni *m* = (tested latents) × (all types),
> distinct from the stratified per-stratum *m*. If (c) is wrong, T5's naive-passer count — the headline
> number of the whole methodological argument — is wrong.

**Cost**: minutes, CPU (operates on count matrices).

---

### M8. Permutation null

**Inputs**: M6 histograms, M3 labels.

**Computation**: for each (arm, layer, stratum, type), **1,000 label permutations** — shuffle PTM labels across residues *within the stratum*, holding the positive count fixed. Recompute the full statistic vector each time.

*Why label-shuffling*: it preserves everything about the latent (firing rate, activation distribution, sparsity) and destroys only the label association. The null therefore captures "what does *this specific latent* achieve by chance." Because stratification means shuffling happens within a residue-identity stratum, residue identity is preserved automatically — the permutation cannot accidentally manufacture a lysine-detector artefact.

*Why this replaces imported thresholds*: VF's bands (Stage 1 pass at AUROC ≥ 0.99) were calibrated on **protein-level** discrimination between β-lactamase classes — ~25% positive rate, family-defining signal. This task is **residue-level** at ~0.5–2% positive rate with local-microenvironment signal. A genuinely excellent succinylation latent might score AUROC 0.85. Importing 0.99 would fail everything, and reporting "no latent passes" against a borrowed threshold is a category error dressed as rigour.

*Four things this buys*: (1) thresholds become data-derived — "pass" = above the 95th percentile of *this* null; (2) AUPRC gets its correct per-type baseline, since chance level *is* the positive rate and varies from ~1% (succinylation) to ~40% (phosphorylation); (3) permutation FDR appears free as a third correction; (4) Arms A and B become comparable via percentile-against-own-null, neutralising the dictionary-width confound at the validation stage.

*Why it is cheap*: permutation operates on **count matrices, not re-run inference** — shuffle and retally a few-MB array. This is the second major payoff of the histogram design.

*Why 1,000*: resolution 0.001 on permutation p-values, sufficient for a stable 95th/99th percentile tail estimate. More is wasted precision given Bonferroni is primary.

**The full set of null variants.** One null is generated per **cell**, where a cell is
(arm, layer, stratum, PTM type, background, negative tier). Enumerated:

| # | Background | Negative population | Purpose |
|---|---|---|---|
| 1 | stratified | full stratum | Thresholds for the significance gate and every validation check |
| 2 | stratified | **hard tier only** | The negative-tier gate (CF-11) — without this it is uncomputable |
| 3 | naive | all corpus residues | Like-for-like passer counts in the naive-vs-stratified comparison |
| 4 | qPTM-stratified | full stratum | Replication thresholds (CF-26) |
| 5 | per OOD species | that species' stratum | Species-specific replication thresholds (CF-14) |

Variants 1–4 run on the human corpus; variant 5 runs once per species-type pair that clears the
applicability matrix (CF-22). The gold tier is **not** nulled — with ~828 sequons for one type the tail
estimate is unstable; it is reported descriptively.

**Edge cases that must be handled explicitly:**

| Condition | Handling |
|---|---|
| Positives < 20 in the cell | Null tail is unstable; mark the cell `null_unreliable` and report the analytic p-value only |
| Hard-tier negatives < positives | Cannot shuffle meaningfully; skip variant 2, report the negative-tier gate as `not_applicable` |
| Latent has zero firings on modified residues | Three effect sizes undefined (CF-19); exclude from the null, count separately |
| Type absent from qPTM or from a species | Skip variants 4/5 for that cell, report `not_applicable` — never a failure |
| All 1,000 shuffles produce identical values | Degenerate (usually a near-dead latent); percentile undefined, mark and exclude |

**Emits**: `null_distributions/*.npy`; `thresholds.json` (95th/99th percentile per metric **per cell as
defined above**), each tagged `exact` or `decile_approx` per CF-27.

> **Three distinct nulls exist in this pipeline. They are not interchangeable and must not share code.**
>
> | Null | What is randomised | Where used | Draws |
> |---|---|---|---|
> | **Label-shuffle null** | Which residues carry the PTM label — hypergeometric resample (CF-27) | All effect-size thresholds; false-discovery estimate | 1,000 |
> | **Position-shuffle null** | Activation values across positions **within a protein** | Structural-vs-sequential clustering **only** | 5 per protein, 100 proteins per latent |
> | **Control-latent null** | Nothing is randomised — the null is an empirical spread over *other latents* and non-SAE directions | Causal z-scores | 20 per control tier |
>
> A frequent confusion: the structural-vs-sequential check does **not** use the label-shuffle null. It asks
> whether a latent's activations cluster in sequence or in 3D space, which does not involve PTM labels at
> all, so its null permutes positions within a protein instead.

> **⚠ CRITICAL SPECIFICATION — CF-27: HOW the label-shuffle null is computed from histograms.**
> The plan asserts the null "operates on count matrices, not re-run inference." True, but the **mechanism
> was never stated**, and the obvious reading — "shuffle labels across residues, then re-tally per-latent
> counts" — **is impossible from histograms**, which have already aggregated away which residue produced
> which activation. An implementor following the plan literally will get stuck or invent something wrong.
>
> **The correct mechanism.** Under the null the positive set is a random subset of the comparison
> population. So for latent $i$, type $\ell$, negative tier $t$:
>
> 1. $H^{\text{pop}}_i[b]$ = that latent's activation histogram over the **whole comparison population**
>    (positives + that tier's negatives). Already stored.
> 2. $P$ = number of positives for that type in that population.
> 3. **One permutation** = draw a **multivariate hypergeometric** sample of size $P$ from
>    $H^{\text{pop}}_i$ across its 64 bins. That is the null positive histogram; the remainder is the null
>    negative histogram. **Drawing $P$ residues at random from the population *is* the shuffle** — no
>    per-residue data is required.
> 4. Recompute every statistic from the two null histograms. Repeat 1,000×; take percentiles.
>
> **The occurrence null is analytic.** Fisher's exact test *is* the hypergeometric test, so for fold
> enrichment and selectivity the permutation distribution and Fisher's null are the same object. The
> permutation machinery earns its cost on **Cliff's delta and AUPRC**, which have no closed form here.
>
> **Cost control — a required two-tier scheme.** A per-latent null for every latent costs
> 1,000 draws × 4,096 latents × ~25 type-tier cells × 21 arm-layer combinations ≈ **2×10⁹ draws**. Too slow.
> The null depends almost entirely on the latent's total firing count and on $P$, so:
>
> | Scope | Scheme | Cost |
> |---|---|---|
> | **Screening** (all tested latents) | Bin latents into **firing-rate deciles** within each (arm, layer, stratum). One null per (decile, type, tier), assigned to every latent in that decile | 10 × types × tiers per arm-layer — seconds |
> | **Candidates** (selected top latents) | **Exact per-latent null**, 1,000 draws each | a few thousand nulls total |
>
> Tag every threshold `exact` or `decile_approx`. **A candidate's reported percentile must always come from
> its exact null**, never the decile approximation.
>
> **Edge cases, all requiring explicit handling:**
>
> | Case | Handling |
> |---|---|
> | $P = 0$ | Skip the cell; record `no_positives` |
> | $P < 10$ | Compute but flag `unstable_null` — a 95th percentile from draws of <10 items is coarse |
> | Latent fires on 0 residues in the population | Statistics undefined; already removed by the activity filter |
> | Latent fires on fewer residues than $P$ | Legal — the null simply concentrates near zero overlap |
> | Population smaller than $P$ | Impossible; indicates a label-construction bug — assert, don't clamp |
> | All firings in one bin | Null is degenerate but valid; Cliff's delta ≈ 0 by construction |
>
> **How many null distributions.** Dimensions are (arm × layer) × background × stratum × type × tier. With
> 21 arm-layer combinations, ~25 types across 6 strata, 2 negative tiers for the stratified background and 1
> for naive: roughly **1,050 stratified + 525 naive ≈ 1,575 cells** — at decile resolution for screening,
> plus exact nulls for candidates only.

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-11: M8 must also produce Hard-tier nulls.**
> CF-6 redefines G4 as "Bonferroni-significant **and above the 95th percentile of the permutation null**
> when tested against **Hard negatives alone**." That percentile does not currently exist: M8 is specified
> per (checkpoint, layer, stratum, type) with the full stratum as the negative population, so there is no
> null distribution for the Hard-tier negative set. **CF-6 is uncomputable without this.**
>
> **Fix**: add a **negative-tier dimension** to the null. Run the same label-shuffle machinery a second
> time with the negative population restricted to Hard-tier residues (Corpus A: unannotated target residues
> in proteins known to carry that type). Cost is the same shuffle-and-retally on a smaller count matrix —
> minutes. The Gold tier (N-glycosylation only, ~828 sequons) is optional; with n that small, report G4
> against Hard and treat Gold descriptively.
>
> **Fallback if M8's structure makes the tier dimension invasive**: drop the null clause and define G4 as
> Bonferroni-significance against Hard negatives alone. Weaker and inconsistent with R6, but still correct —
> and strictly better than the borrowed 0.50 it replaces.

> **Scope — run for both backgrounds.** The null is generated separately for the **stratified** and **naive** label assignments. The stratified null supports G1 and every validation threshold. The **naive null is required for M13**: comparing "Bonferroni-passers under naive" against "Bonferroni-passers under stratified" is only a like-for-like comparison if both counts are produced by the *same* criterion, and G1 is "Bonferroni-significant **and** above the 95th null percentile." Using raw p-values for one arm of the comparison and the full criterion for the other would understate the naive arm's false-positive count and weaken the very result M13 exists to establish. Naive permutation shuffles across *all* residues in the corpus, since the naive analysis has no stratum — which is precisely why residue-identity latents survive it, and why the contrast is informative.

**Cost**: minutes to ~1 h, CPU.

---

### M9. Correction and candidate selection

**Inputs**: M7, M8.

**Computation**:
1. **Bonferroni primary**, m = (tested latents surviving the activity filter) × (types in that stratum), **per checkpoint per layer**. For the **naive** background there is no stratum, so m = (tested latents) × (all types tested). This is the larger correction burden, which is correct — the naive analysis tests more hypotheses because it does not restrict by residue chemistry.
2. Benjamini–Hochberg and permutation FDR as secondaries.
3. Apply gates:

| Gate | Requirement | Failure consequence |
|---|---|---|
| **G0** | Arm did not trip the broken-instrument floor (CF-25) | Arm excluded from all claims |
| **G1** | Bonferroni-significant occurrence **and** enrichment above the 95th null percentile | Not a candidate |
| **G2** | **Naive background only** — residue-dominance <7/10, from M6's amino-acid firing matrix and top-20 heap. **Not applied to stratified candidates** (see CF-13) | Hard exclusion **of naive passers**, feeding T5/F4 as demonstrated false positives |
| **G3** | V1 top-type/second-type enrichment ratio ≥ 1.5 | **Relabelled** stratum-general, not discarded |
| **G4** | V5 enrichment survives against Hard negatives | Retained with annotation-bias caveat |
| **G5** | V6 retains ≥50% enrichment when crosstalk excluded | Flagged possibly co-occurrence-driven |

> **⚠ CRITICAL CORRECTION — CF-13: residue-dominance is TAUTOLOGICAL for stratified candidates and would
> hard-exclude every true positive.**
> VF's Stage 4 criterion (≥7 of the top 10 activating positions sharing one amino acid → automatic fail)
> was designed to catch latents that are secretly amino-acid detectors — their L6/2417 fired 10/10 on
> alanine while posing as a β-lactamase class feature.
>
> **Under the residue-stratified design that logic inverts.** Every succinylation site *is* a lysine.
> A perfect succinylation detector fires on lysines, so its top-10 activating positions will be **10/10
> lysine** — and G2, applied as written, would **hard-exclude it**. The better the latent, the more
> certainly it fails. This is the same class of error as CF-10: a criterion transplanted from a source
> where the confounding property was independent of the label, into a setting where the confounding
> property is *definitional* to the label.
>
> **Why no replacement gate is needed.** G2's purpose — distinguishing a generic lysine detector from a
> succinylation detector — is **already served by construction** by the stratified background (§0.4). A
> pure lysine detector scores exactly FE = 1.00 within the lysine stratum. That is the whole argument for
> stratification, and it makes an additional dominance gate redundant rather than merely awkward.
>
> **Corrected specification, three parts:**
> - **Naive background: keep ≥7/10 exactly as written.** There it does its original job — identifying the
>   residue-identity detectors that inflate naive enrichment. This is what colours F4's scatter and fills
>   T5's "of which residue-dominant" column. **CF-13 does not weaken the headline methodological result.**
> - **Stratified candidates: G2 is dropped as a gate.** Dominance computed within a stratum is trivially
>   10/10 for every latent, since all stratum residues are the same amino acid — it carries no information.
> - **Replace it with a reported diagnostic, not a gate**: *target-residue firing preference* — the fraction
>   of the latent's **corpus-wide** firings landing on the stratum's target residue, divided by that
>   residue's background frequency. A succinylation latent firing 60% on lysines (background 5.8%) is
>   chemistry-aware; one firing 6% on lysines while still showing within-stratum enrichment is a generic
>   context detector whose enrichment may be incidental. Informative, and computed free from M6's
>   amino-acid matrix — but reported in T6, not used to exclude.
>
> **Consequence for the promotion rule**: **G0 becomes the only hard exclusion for stratified candidates.**
> G2 remains hard for naive passers. State this asymmetry explicitly in Methods rather than letting a
> reader assume one rule applies to both analyses.

**G0 is the only hard exclusion for stratified candidates** (G2 is hard for naive passers only, per CF-13).
Everything else *relabels*. This is deliberate: VF's binary framing produced "no node passes," which is true but unhelpful. A latent failing G3 is genuinely informative — it says ESM-2 represents "modifiable lysine context" — and discarding it would throw away a finding.

4. Select **top 20 per type per arm** by null percentile of enrichment, among G1+G2 passers — **computed on the `discovery` partition only.**

> **Why discovery-only selection.** Taking the maximum enrichment over ~1,200 tested latents is taking the maximum of a noisy quantity, which is optimistically biased. Reporting that same enrichment as the result would be circular. Selecting on discovery and re-estimating on holdout (V12b, M19) makes the reported estimate unbiased, and the discovery-minus-holdout gap is itself a reportable measure of selection optimism.
>
> **Note**: the *main* statistical results (T4) are computed on the **full corpus** — discovery and holdout histograms summed. The split governs **candidate selection only**. The two are not in conflict: T4 answers "which latents are associated with which PTMs across the whole corpus," while V12b answers "is the enrichment of the *specifically chosen* candidates inflated by having chosen them."

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-2: candidate scope is the STRATIFIED background only.**
> The plan never stated this explicitly, and the implementation consequently let **naive-background
> candidates flow into Pass 2 and the validation cascade** — visible in changelog entry #16, where
> `v12b_holdout_reestimation()` had to be taught to handle a candidate whose `stratum` is the
> pseudo-stratum `"ALL"`.
>
> **That fix was technically correct but treated a symptom.** The naive background exists for exactly
> one purpose: M13's false-positive quantification (T5, F4). Its passers are, by construction, the
> population expected to contain residue-identity detectors — spending Pass 2 storage, structural
> grounding, and causal compute on them is wasted effort on latents already known to be confounded.
>
> **Correction**: `candidates.json` (M9) contains **stratified-background candidates only**. Naive-background
> results stay in the M7/M9 statistics tables and feed M13, and their G2 residue-dominance verdicts come
> from M6's amino-acid firing matrix — no Pass 2 data is required for anything M13 does. This removes
> the `stratum == "ALL"` case from every downstream V-check rather than special-casing it, and cuts
> Pass 2 and Phase VI cost.
>
> **Keep entry #16's fix anyway** as defensive code, but it should now be unreachable.

> **Arm C scope — full cascade.** Arm C (ESM-2-8M) participates in **all phases**, including Pass 2 (M15), the Tier 1–4 validation cascade (M16–M20), and the causal phase (M21–M24). It was briefly scoped to Phases I–IV on compute grounds; the arithmetic does not support that, since ESM-2-8M forward passes are roughly 80× cheaper than 650M and the full cascade on C adds ~5 min of Pass 2 and under an hour of causal work.
>
> **Why full inclusion is scientifically necessary, not merely affordable.** Three contrasts exist only if C is validated: (i) **does scale change *what kind* of PTM latent exists**, not just how many — if C's latents fail G3 (type specificity) more often than B's, small models encode "modifiable context" without modification *type*, a substantive claim about representational granularity; (ii) **C-vs-B isolates scale for feature absorption (V10)**, since B and C share architecture (L1) and dictionary width (10,240), making it a cleaner absorption contrast than A-vs-B, which confounds scale with architecture; (iii) **ESM-2-8M is the exact checkpoint VF validated**, so running the full cascade on C makes every number directly comparable to their published results and makes this work a strict superset of their setting rather than a parallel one.
>
> **Anchors for C are depth-matched, not arbitrary**: layer 24 of 33 is relative depth 0.73, and 0.73 × 6 ≈ **layer 4** — which is independently VF's structurally strongest ESM-2-8M layer. Layer 6 is terminal, matching layer 33's negative-control role.

**Emits**: `results_corrected.parquet`; `candidates.json`; **T4 — Core statistical results** (rows = latent × type × arm × layer; columns grouped occurrence / magnitude / discrimination / secondary, each with null percentile per R1 and R6).

**Cost**: minutes, CPU.

---

## Phase IV — Analysis from histograms (CPU, no further inference)

*Everything in M10–M14 runs on stored histograms. No GPU.*

### M10. τ-stability

**Computation**: for all nine τ values (free — suffix sums), recompute the Bonferroni-passing set. Report **Jaccard overlap** with the τ = 0 set.

*Why*: directly answers "did you tune the threshold?" A stable set means the result is threshold-independent; a rapidly shrinking set means it is fragile and must be reported as such. Precedent for sweeping: VG&A {0.01, 0.10, 1.00}; InterPLM {0, 0.15, 0.5, 0.6, 0.8}; InterProt 0.1–0.9; SAEBench recommends a range.

*Scope*: **computed everywhere, reported at anchors** — layers 24 and 33 (Arms A, B) plus the empirical peak in main text; full τ × layer × type grid in supplementary.

**Emits**: **F6b** (Jaccard vs τ); **S4** (full grid).

---

### M11. Layer and architecture analysis

**Computation**:
1. PTM-enriched latent **rate** vs layer, per arm (R3 — rate, not count).
2. **Enrichment among passers** vs layer — the number-vs-strength companion. *SPIRAL Table 2 shows these dissociate: layer 0 had 97.5% passing at 1.04× enrichment; layer 5 had 44.3% at 1.61×.*
3. Identify the empirical peak layer per arm.
4. Arm A vs Arm B at layer 24 — paired per-type comparison of rates and enrichments.
5. Layer 33 (both arms) as the built-in negative control.
6. Compare the peak against F3c's ID plateau.

*Prior from four papers and two modalities*: family/type-general structure peaks early-to-middle and declines late. SPIRAL layer 5/12 (RNA); InterProt early-to-mid of 33; VF layer 4/6 structurally strongest, layer 6 causally strongest but concept-shared; VG&A layer 3/6 by ID plateau; Gujral layers 6–10/12.

*Caveat for Methods*: layer 24 was chosen by **checkpoint availability, not biology**, and sits past InterProt's reported family-specificity peak. The sweep reveals whether that was fortunate. PTM signal peaking at a different depth than family signal would itself be a result.

**Emits**: **F5a** (rate vs layer), **F5b** (enrichment among passers vs layer), **F6a** (A vs B at 24).

---

### M12. Core statistical figures

**Emits**:
- **F5c** — selectivity histogram with the 1/L chance reference line. *SPIRAL Fig 1b. Critical companion: selectivity alone is misleading under imbalance — with Stem at 46.4%, a random feature scores 0.46. Always paired with enrichment.*
- **F5d** — Cliff's δ histogram, paired in-caption with the significance pass rate (R1).
- **F5e** — **occurrence vs magnitude scatter**: fold enrichment on x, Cliff's δ on y. **Novel.** Four interpretable quadrants: high/high (strong detectors), high-occurrence/low-magnitude (broad weak), low-occurrence/high-magnitude (rare saturating), low/low (marginal).
- **F5f** — dual-ranked bars (R2): top 10 by selectivity, top 10 by enrichment, each annotated with the other metric.
- **S1** — activation-frequency vs activation-extent landscape.

---

### M13. Naive versus stratified — the headline methodological result

**Computation**:
1. Bonferroni-passer counts under each background, per type.
2. Latents passing naive but failing stratified; of those, how many trigger residue-dominance (≥7/10).
3. **Latents passing naive whose association is partly supported by chemically impossible residues** (M7 step 7) — e.g. a "succinylation-enriched" latent whose firings on succinylation positives include non-lysine residues.
4. Median FE inflation factor; compare against the **pre-stated prediction** 1/*f*<sub>residue</sub> (≈17× lysine, ≈28× asparagine, ≈7× Ser/Thr).

*Why the chemical-impossibility counter is worth reporting separately from residue-dominance* **(adapted from COMPASS-PTM)**. They observe a baseline predictor (SAGEPhos) assigning acetylation and ubiquitination — both lysine-exclusive — to a **threonine** residue, with "biochemically impossible and misleadingly high probabilities." The same failure mode is available to an enrichment analysis run on an unstratified background, and it is a *different* diagnostic from residue-dominance: dominance asks whether a latent is an amino-acid detector; chemical impossibility asks whether the *association itself* violates known chemistry. A latent can pass dominance (its top-10 positions are mixed) while still drawing its succinylation signal from serines and glutamates. Reporting both gives two independent, chemistry-grounded false-positive counts, and neither requires any computation beyond what M7 already produces.

*Why this is not a straw man*: the naive analysis is what InterPLM's `ft_mod_res`/`ft_carbohyd` F1, VG&A's glycosylation precision/recall, and ESMC-SAE's MI ranking all effectively do. Reporting the difference converts a methodological argument into a measured false-discovery count.

*Why stating the prediction in advance matters*: checking a pre-stated prediction is stronger evidence than reporting the number afterwards.

**Emits**:
- **T5 — Naive vs stratified**: rows = type; columns = passers each background, naive-pass/stratified-fail, of which residue-dominant, **of which chemically impossible**, median inflation, predicted inflation, observed/predicted ratio
- **F4** — (a) paired scatter FE(naive) vs FE(stratified), coloured by residue-dominance pass/fail, **marker shape** distinguishing chemically-impossible associations, y = x diagonal marked; (b) passer counts per background with R7 annotation; (c) observed vs predicted inflation per stratum

*Points far above the diagonal in F4a that also fail dominance are **individually identifiable false positives**, not a statistical abstraction. This is the figure that converts the argument into evidence.*

---

### M14. Sequence-level track

**Inputs**: M6 (requires per-protein mean activations — accumulate alongside histograms in M6).

**Computation**:
1. $v_q = \frac{1}{T_q}\sum_t f_{qt}$ (SPIRAL Eq. 12).
2. Pre-filter: **latent fires (f > 0) on ≥1 residue in ≥10 sequences.** (SPIRAL's magnitude clause dropped — CF-12.)

> **⚠ CORRECTION — CF-12: SPIRAL's 0.05 sequence-mean floor would eliminate nearly every latent.**
> The threshold is calibrated to SPIRAL's activation density. Their SAEs run at L₀ ≈ 12.6% of the
> dictionary — each latent fires on roughly 12.6% of tokens — so a sequence-mean near 0.05 is a reasonable
> "meaningfully active" bar *there*.
>
> **Arm A is ~8× sparser by construction**: TopK with k = 64 of 4,096 means each latent fires on ~1.56% of
> residues. A latent firing on 1.56% of residues at mean strength 0.5 has a sequence-mean of ~0.008 —
> **six times below the transplanted threshold.** Applying 0.05 would filter out essentially the whole Arm A
> dictionary before any test ran, and the sequence-level track would silently return nothing.
>
> **Fix**: keep the coverage clause, drop the magnitude clause. This preserves SPIRAL's actual intent
> (exclude latents with too little data for a stable per-protein mean) without importing a density
> assumption that does not hold across architectures. If a magnitude floor is wanted later, derive it from
> the observed per-arm sequence-mean distribution — never from a constant borrowed across SAEs.
3. **Rank-residualise against sequence length** before testing (Eq. 15):
   $$r_{iq}^\perp = \tilde r_{iq} - \hat\beta_i\tilde\ell_q,\qquad \hat\beta_i = \frac{\sum_q \tilde r_{iq}\tilde\ell_q}{\sum_q \tilde\ell_q^2}$$
4. **One-vs-rest Mann-Whitney per PTM type**: proteins carrying type X vs proteins not carrying it, on the rank-residualised pooled value. Effect size: Cliff's δ.

> **Correction to the naive SPIRAL transplant.** SPIRAL uses Kruskal-Wallis across mutually exclusive RNA type classes — an RNA is one type. **Proteins are not**: a single protein routinely carries phosphorylation, acetylation, ubiquitination and glycosylation simultaneously. Applying a multi-class Kruskal-Wallis across "protein classes" would repeat at protein level exactly the non-mutual-exclusivity error that §3.4 corrects at residue level. One-vs-rest Mann-Whitney is the protein-level analogue of the residue-level primary and handles overlap natively. **Kruskal-Wallis + η² (Eqs. 13–14) is retained as an optional secondary**, restricted to singly-modified proteins where a partition genuinely exists — but note this subpopulation is far smaller at protein level than at residue level, so it will often be underpowered and may be skipped.

5. **Protein-level event coherence** *(adapted from COMPASS-PTM)*: for each candidate latent and each protein carrying ≥1 site of its preferred type, compute the Jaccard overlap between {residues where the latent fires} and {residues carrying that type}, then average across proteins. COMPASS-PTM report a mean protein-level event-F1 of 0.6430 as evidence that predicted residue–PTM events "align with the ground-truth program at the level of whole proteins."

*Why add this*: residue-level enrichment can be high while a latent's firing pattern *within* any individual protein is incoherent — firing on three of a protein's ten succinylation sites plus twelve unmodified lysines. Averaged over the corpus this still yields enrichment; per protein it is not a coherent PTM program. This measure is the coherence check one level above the residue. It does **not** mean-pool — it compares two sets of residue positions — so InterProt's mean-pooling caveat does not apply to it. Requires M15's per-residue positions, so it runs after Pass 2 even though it belongs conceptually to this track.

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-4: M14 step 4 test change, and S9's residue universe.**
>
> **(a) Step 4 was built as a multi-class Kruskal-Wallis and must become one-vs-rest Mann-Whitney.**
> The changelog's Phase IV section records only a matplotlib tick fix, so the statistical test almost
> certainly still follows the superseded specification. Proteins are not mutually exclusive across PTM
> types — a single protein routinely carries phosphorylation, acetylation, ubiquitination and
> glycosylation at once — so a multi-class *H* statistic over overlapping "protein classes" is not
> well-defined. **Reuse the Mann-Whitney + Cliff's δ implementation already built for M7 Part 2**,
> applied to rank-residualised pooled per-protein values. This is a small change: the same function,
> different input.
>
> **(b) S9's residue universe must be disclosed, and the COMPASS-PTM comparison dropped.**
> Changelog entry #27 discloses that "residues where the latent fires" is evaluated over M15's *stored*
> residues (positives + a 5× negative sample), not every residue in the protein. That is the only
> option available — M15 deliberately does not store all residues — but it has a consequence the entry
> does not draw out: the universe is ~6× the positive count rather than the full stratum, so **the
> Jaccard denominator excludes most true negatives and the metric is inflated by construction.** The
> reported 0.45–0.83 range reflects the sampling design as much as the latents.
>
> Three requirements follow. **(i)** Compute S9 for the activity-matched control latents over the
> *identical* stored-residue universe, and report candidate-minus-control, never the raw value alone.
> **(ii)** Report the sampling ratio (stored residues ÷ stratum residues per protein) alongside every
> S9 value. **(iii)** **Do not compare S9 to COMPASS-PTM's 0.6430.** Their protein-level event-F1 was
> computed over all residues of a protein; a 5×-negative-subsampled Jaccard is a different quantity on
> a different denominator, and presenting them side by side would be an invalid comparison. Cite their
> figure as motivation for the *concept*, not as a benchmark for the number.

> **Interpretation ceiling — report relatively, never absolutely.** Two effects bound this measure from above independently of latent quality: **annotation incompleteness** (unannotated true sites are counted as false positives, and N-GlyDE's 65.1% sequon-occupancy figure shows how large this can be), and **τ-sensitivity** (at τ = 0 a TopK latent fires on many residues, mechanically depressing Jaccard). An absolute value of 0.3 is therefore uninterpretable on its own. Report it only **relative to the activity-matched control latents stored in M15** and **across candidates**, and state the ceiling explicitly in the caption.

*Why length control here and not at residue level*: mean-pooling makes any position-bounded latent automatically length-sensitive (a latent firing near the N-terminus has mean activation inversely proportional to length). SPIRAL found PCA components correlating with length at |r| up to 0.74. At residue level the test already conditions on individual residues.

*Why secondary*: InterProt's own limitations flag mean-pooling as known-weak (citing NaderiAlizadeh & Singh 2024). The residue-level track is the primary claim.

**Emits**: **S8** (sequence-level one-vs-rest Mann-Whitney, Cliff's δ, length-confound diagnostic — available immediately after M6); **S9** (protein-level event coherence — runs after M15, since it needs per-residue positions).

> **Ordering note.** M14 is listed in Phase IV because its primary content depends only on M6. Its step 5 (protein-level event coherence) is the one exception and executes after M15. An agent should implement M14 steps 1–4 in Phase IV and defer step 5 to Phase V.

---

## Phase V — Targeted analysis (GPU)

### M15. Pass 2 — targeted activation store

**Inputs**: M9 candidates.

**Computation**: re-run inference storing sparse per-residue activations, restricted on **three axes**:

| Axis | Restriction | Why |
|---|---|---|
| Layer/checkpoint | **Anchors only** — layer 24 (A, B), layer 33 (A, B), layer 4 and 6 (C), plus the empirical peak per arm ≈ **9 combinations, not 21** | Screening across all layers already happened in M6 |
| Latents | The **20 candidates per type per arm** selected in M9, **plus 20 random control latents and 20 activity-matched control latents** (needed by M24's three-tier baselines) = **60 per type per combination** | Storing controls now avoids a second targeted pass in Phase VI |
| Residues | PTM-positive + a stratified random sample of negatives at **5× the positive count** | Per-residue analyses need positives and a matched negative sample, not the full background |
| **Profile sample** | Separately, **full sparse latent vectors** for a stratified random sample of ~50,000 residues per combination | Required by V12c's kNN utility probe, which needs whole latent profiles rather than selected latents. Sparse storage (TopK: exactly 64 non-zeros; L1: ~L₀ non-zeros) keeps this at ~26 MB per combination |

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-5: `anchor_layer_set()` must iterate all three arms.**
> Changelog entries #15 and #25 record that `anchor_layer_set()` hard-codes its loop over `["A", "B"]`,
> and #25 explicitly documents this as deliberate, citing the plan's own M9 scope note that Arm C
> "does not enter Pass 2." **The implementor followed the plan correctly — the plan was subsequently
> changed and this is the resulting stale reference, not an implementation error.**
>
> Arm C now has anchors `[4, 6]` and participates in the full cascade (M9 scope note, revised). Required
> changes: **(i)** iterate `ARMS` rather than a hard-coded list, keeping entry #15's empty-layers guard
> before the `ARMS[arm]` lookup so an arm absent from a test fixture still skips cleanly; **(ii)** update
> entry #25's docstring, which currently cites superseded plan text and would otherwise mislead the next
> auditor into believing the exclusion is still intended; **(iii)** extend the same change to M16–M24, all
> of which key off the anchor set. Cost is negligible — ESM-2-8M forward passes are ~80× cheaper than 650M.

**Must also store residue position and identity** — without them, residue-dominance re-confirmation (V2), sequence logos (V8), and co-firing analysis (M25) are impossible, and those are the reason M15 exists.

**Budget check** (succinylation): 60 latents × 9 combinations × (3,000 positives + 15,000 negatives) × 4 bytes ≈ **39 MB/type**; across ~25 types ≈ **970 MB**, plus ~230 MB of profile samples ≈ **1.2 GB total**. *Unrestricted alternative*: 21 × 500 × 6.5M × 4 B ≈ **273 TB**. The restriction is what makes M15 possible at all.

**Emits**: `targeted_activations/` in `/kaggle/temp`.

**Cost**: ~30–60 min GPU.

---

### M16. Tier 1 — specificity validation

**Inputs**: M15, M6 histograms.

| ID | Check | Method | Failure signature |
|---|---|---|---|
| **V1** | Cross-type within stratum | Enrichment vector across all other types in the stratum; report top/second ratio | Uniform across all K-types → lysine-context detector |
| **V2** | Residue-identity dominance | Of the top 10 activating positions corpus-wide, how many share one amino acid? **≥7/10 → hard fail** | A "PTM latent" that is an amino-acid detector (VF: L6/2417 = 10/10 alanine) |
| **V3** | Family/domain breadth | Partition activating proteins by InterPro/Pfam family; AUPRC per family vs within-family negatives; report the **minimum and the median** over families with ≥30 sites, plus the family count | High overall AUPRC driven by one or two families |
| **V4** | Homology control | Pairwise sequence similarity of co-activating proteins vs a size-matched random set | Elevated similarity → possible homology artefact |
| **V5** | Annotation-tier robustness | Recompute enrichment against Gold / Hard / Background separately | Present only against Background → coverage artefact |
| **V6** | Crosstalk robustness | Recompute excluding co-modified residues | Collapse → co-occurrence leakage from another type |

*V1 replaces rather than supplements VF Stage 2*: VF had to **construct** harder comparison classes (Class C, D β-lactamases). Here the comparison set is given by chemistry, so the test is both harder and more principled, with no arbitrary choice of comparison class.

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-6: G4 needs its own criterion, not V6's borrowed 0.50.**
> Changelog entry #21 records that `v5_three_tier()` resolves `G4_status` using
> `cfg.v6_crosstalk_retention_min` (0.50), described as "this implementation's own consistent choice,
> since the plan gives V6 an explicit 0.50 bar but only a qualitative failure signature for V5."
>
> **The gap in the plan was real and the stopgap was reasonable, but the two quantities are not
> commensurable.** V6's 0.50 is a *retention ratio under a subset restriction* — how much enrichment
> survives when crosstalk residues are removed. V5 asks something different: whether enrichment exists
> against studied-but-unmodified residues (Hard tier) rather than only against unstudied ones
> (Background tier). A retention ratio between two differently-constructed negative sets has no
> principled 0.50 bar, and borrowing one silently imports V6's tolerance into an unrelated decision.
>
> **Correct G4 criterion, requiring no new constant**: the latent **remains Bonferroni-significant with
> fold enrichment above the 95th percentile of the permutation null when tested against Hard negatives
> alone.** This is G1's own criterion re-applied to a different negative set, which is exactly the right
> shape — "does the association survive when the easy negatives are removed" — and it is consistent with
> every other threshold in the pipeline being null-derived (§0.5, R6) rather than hand-set. Set
> `v5_hard_tier_criterion: "significant_and_above_null"` (§0.2) and leave `v6_crosstalk_retention_min`
> to V6 alone.

*V3 fixes VF's gap*: VF used only the largest 5 of 8+ families. This uses **all** qualifying families. **Report minimum and median together**: the minimum is the strict breadth criterion but is dominated by whichever family sits closest to the 30-site floor and is therefore noisy; the median states whether the latent works broadly. A latent with median 0.4 and minimum 0.05 is broadly useful with one weak family; one with median 0.06 and minimum 0.05 is uniformly weak. The minimum alone cannot distinguish these.

*V4 precedent*: Gujral found co-activating proteins **do** have higher similarity, but its correlation with interpretability was weak (r = 0.117–0.263).

**Emits**: validation columns V1–V6 for T6; **S3** (per-family AUPRC heatmap).

---

### M17. Tier 2 — mechanistic grounding

**Inputs**: M15; AlphaFold structures for corpus proteins.

| ID | Check | Method |
|---|---|---|
| **V7** | Structural vs sequential clustering | Mean activation within ±2 sequence positions (sequential) vs mean activation of residues within 6 Å in 3D (structural); permutation null (5 permutations/protein, 100 proteins/feature); paired *t*-test + Cohen's *d*; **Bonferroni-corrected**. **This null is not the label-shuffle null.** The permutation null of the statistics phase shuffles *PTM
labels across residues*. This one shuffles *activation values across positions within a single protein*,
5 permutations per protein, asking whether the observed spatial clustering exceeds what a random
rearrangement of the same activations produces. Different question, different machinery, computed here —
do not reuse `thresholds.json`. **Exclude latents with <25 qualifying proteins** (InterPLM's own minimum) and report exclusions per type — for rare types with sparse AlphaFold coverage this may exclude most candidates, which is itself reportable. *InterPLM §5.4.2.* |
| **V8** | Motif recovery | Sequence logo built from the **top 100 highest-activating positions** for the latent (COMPASS-PTM's protocol), spanning **P−7 to P+7** with **P0 = the modified residue**, compared against the known consensus |
| **V9** | Solvent accessibility & structural context | Spearman ρ between activation and relative solvent accessibility; secondary-structure composition of activating positions. *VF Stage 4 adapted — distance-to-active-site replaced by accessibility, the mechanistically relevant variable for PTMs.* |

**V7 is the highest-value item here, and carries a falsifiable pre-stated prediction.** PTM mechanisms split naturally along this axis:
- **N-glycosylation** is defined by a *sequence* motif (N-X-S/T) → predicts **sequential** clustering.
- **O-GlcNAcylation and succinylation** depend on enzyme accessibility and local 3D environment, no strict consensus → predicts **structural** clustering.

Either outcome is informative. If confirmed, this is a mechanistic finding about how ESM-2 encodes different PTM classes, not merely a validation check. No paper in the reading list produces this for PTMs.

**V8's strongest possible result**: a glycosylation latent's motif logo independently recovering **N-X-S/T with proline depletion at X** — reconstructing a known biochemical rule from an unsupervised representation, with no supervision on the motif.

**Concrete motif targets, if phosphorylation latents are examined** *(from COMPASS-PTM, which recovered all four from a supervised model's top-100 predictions)*: **AGC** family — basophilic, basic residues at P−3 and P−2, i.e. R-R-X-S/T; **CAMK** — Arg at P−3 plus a large hydrophobic (Leu) at the distal P−5, i.e. L-X-R-X-X-S/T; **CMGC** — proline-directed, near-absolute Pro at P−2 and P+1, i.e. P-X-S/T-P; **TK** — acidophilic upstream residues plus a subtle Pro preference at P+3. These are the published targets a phosphorylation-associated latent's logo would be checked against, and recovering any of them **unsupervised** would be a stronger claim than COMPASS-PTM's, whose logos were built from a model trained on PTM labels.

**Emits**: **F6c** (structural vs sequential scatter, coloured by type, diagonal marked); **F6d** (sequence logo); **S6** (structure renderings, top 3 latents — for missing-annotation cases use InterPLM Fig 6's convention: pink/dark for annotated, **green for strongly-activating-but-unannotated**, captioned as implied pending confirmation); **S7** (accessibility correlations).

---

### M18. Tier 3 — representational integrity

**Inputs**: M15; raw ESM-2 activations at anchor layers.

| ID | Check | Method |
|---|---|---|
| **V10** | **Feature absorption** | **Precondition (CF-15): report the probe's own AUPRC vs its baseline first.** Train a linear probe on raw ESM-2 activations for PTM-type presence (ground-truth direction). Identify the primary latent via k-sparse probing. On true positives where the primary latent is silent but the probe succeeds, check whether a *different* latent fires **and** aligns with the probe direction (high cosine). |
| **V11** | **Feature splitting** | k-sparse probing sweep, k ∈ {1, 2, 3, 5, 10}; report the F1 gain curve |

> **⚠ CORRECTION — CF-15: the absorption test is uninterpretable if its own ground-truth probe is weak.**
> SAEBench's protocol was built on a first-letter spelling task, where the concept is *deterministic* given
> the token — the raw-activation probe is near-perfect, so "probe succeeds but main latent fails" is a clean
> set. **PTM presence is not deterministic from a residue embedding**; predicting it from raw ESM-2
> activations is hard, and the probe may sit close to its AUPRC baseline.
>
> If the probe is near chance, that failure set is mostly probe noise and any latent co-firing there looks
> like an absorber. **Report the probe's AUPRC against its positive-rate baseline before reporting any
> absorption result**, and treat absorption as uninterpretable for any PTM type where the probe does not
> clear its baseline by a stated margin. A precondition, not a caveat.

*V10 matters disproportionately for Arm A*: TopK is flagged absorption-prone by InterProt's own authors (citing Karvonen et al.) because hard top-k competition gives narrow correlated latents an incentive to outcompete a broader one. **Absorption in Arm A but not Arm B at layer 24 is a directly attributable architecture effect** — exactly what the A-vs-B contrast was designed to isolate, and a contribution to the TopK-vs-L1 question independent of PTM biology.

*V11 changes how results are reported, not just whether they pass*: a type whose best single latent (k=1) achieves near-ceiling classification is a categorically stronger finding than one requiring k=5 jointly. Both are reportable; conflating them is not.

**Emits**: V10, V11 columns for T6.

---

### M19. Tier 4 — generalisation and utility

| ID | Check | Method | Needs |
|---|---|---|---|
| **V12a** | Cross-species (OOD) | Same latent index, same type, in mouse / rat / *S. cerevisiae* / *E. coli*. **Re-run the full three-part decomposition on the OOD corpus**; report whether enrichment replicates against an **OOD-specific** permutation null, plus enrichment ratio OOD ÷ human. **Not** VF's activation ratio — CF-14. | **New corpus + new inference** — repeat M2/M3/M6 for the OOD species |
| **V12b** | Held-out re-estimation | For each candidate selected in M9 on the `discovery` partition, recompute occurrence, enrichment, selectivity, Cliff's δ, and AUPRC on the **`holdout` partition** (M2 split, cluster-level, 20%, seed 42). Report **discovery value, holdout value, and their ratio** | M6 holdout histograms — **no new inference** |
| **V12c** | kNN utility probe | Balanced accuracy of kNN on SAE latent profiles predicting PTM type, vs **length-only** and **amino-acid-composition-only** baselines (SPIRAL Eq. 16). Additionally report **macro-averaged F1 and MCC**, the metrics COMPASS-PTM and the wider PTM-prediction literature use, so this probe is quotable against published predictors | M15 profile sample |

*Data availability confirmed*: all four OOD species have prepared CPLM files — `Mus musculus.zip`, `Rattus norvegicus.zip`, `Saccharomyces cerevisiae (strain ATCC 204508 or S288c).zip`, `Escherichia coli (strain K12).zip`. No substitution needed; the lysine stratum transfers directly.

> **⚠ CORRECTION — CF-14: VF's Stage 6 statistic has the OPPOSITE polarity to what this check needs.**
> VF's metagenome scan tests **specificity** — a good β-lactamase node should *not* fire on unrelated
> metagenome sequences, so they pass at activation **ratio < 0.10** with zero sequences exceeding the 95th
> percentile. Low firing = good.
>
> **This check tests the opposite property.** A succinylation latent *should* fire on *E. coli* succinylation
> sites; that is what generalisation means. Carrying VF's thresholds over would mark every successfully
> generalising latent as a failure and reward latents that ignore the OOD data. Compounding it, homologous
> proteins across species are far more similar than metagenome-vs-β-lactamase, so activation ratios sit near
> 1.0 regardless of latent quality — uninformative even setting polarity aside.
>
> **Fix**: drop the activation-ratio statistic entirely. Run the same occurrence / magnitude / discrimination
> decomposition on the OOD corpus and ask whether enrichment **replicates** — significant, above the
> OOD-specific null percentile, same preferred type — reporting enrichment ratio OOD ÷ human as the
> generalisation measure. **No VF threshold is carried over in any form.**

*V12a's purpose*: distinguishes "this latent encodes succinylation chemistry" from "this latent encodes something about human protein composition that correlates with human succinylation annotation." ***E. coli*** **is the sharp case** — succinylation is well-characterised there and the evolutionary distance makes shared human-specific bias implausible. This is VF's Stage 6 with a biologically interpretable OOD set.

*V12b's purpose*: quantifies **selection optimism**. Candidates were chosen as the maximum-enrichment latents out of ~1,200 tested, so their discovery-partition enrichment is inflated by the act of selection. The holdout re-estimate is unbiased. Report the **retention ratio** = holdout enrichment / discovery enrichment:

| Retention | Reading |
|---|---|
| ≈ 1.0 | Selection optimism negligible; the enrichment is a stable property of the latent |
| 0.7–0.9 | Mild optimism, expected and unremarkable; report the holdout value as the headline |
| < 0.5 | Substantial — the candidate was largely selected on noise. Report both values prominently and weaken the claim accordingly |

Types falling below 200 sites in either partition are excluded from V12b and marked as such; they remain in the main analysis, which uses the full corpus. This is a **free** check — it reuses M6's holdout histograms and requires no additional inference.

*V12c's purpose*: answers "is this superficial?" A representation can encode PTM-relevant information entirely explained by length or composition. Failing to beat both baselines means the enrichment is real but uninteresting.

**Emits**: **F9** (cross-species scatter); **S5** (kNN vs baselines); V12 columns for T6.

**Cost**: ~1 h GPU for the OOD inference.

---

### M20. Validation profile assembly

**Inputs**: M16–M19.

**Computation**: assemble the profile matrix. Cells hold **scores with null percentiles, never verdicts.**

```
Latent      Enrich  Sel    Cliff-δ  AUPRC   V1     V2     V3min  V5(H)  V6    V7        V10   V12a
A-L24/1893   4.2    .71     .38     .19    2.8×   3/10    .14    3.1×   .82   seq***    no    .61
            (99.8) (99.2)  (98.1)  (99.5)                                      d=1.4
B-L24/7204   3.1    .64     .29     .14    1.2×   2/10    .09    2.9×   .91   struct**  yes   .58
            (99.1) (97.4)  (95.2)  (98.3)                                      d=0.7
```

*Why not a verdict*: VF's own results show a latent strong on one axis can be null on another, and that these are **different kinds** of latent, not different qualities of one kind — top-ranked node AUROC 1.000 but **fails** causally; L3/2429 A-vs-B 0.998 but A-vs-C **0.206**; L6/1255 best causal evidence (+3.70 ablation, +8.50 steer) but A-vs-C **0.000**. Collapsing to Pass/Mixed/Fail turns three informative objects into "all failed." A single scalar is computed **for ranking only** and stated as such.

**Emits**: **T6**; **F7** (heatmap + pass-count distribution — converts VF's implicit "no node passes all six" into an explicit quantified distribution).

---

## Phase VI — Causal (GPU)

*Every result so far is correlational. Three independent lines say that is insufficient: VF's top node by discrimination (AUROC 1.000, d = 19.04) **failed** causal validation; H&F state "even if attention maps reflect biologically relevant interactions, it is not guaranteed that this information is used" (supported by pruning 7.9% of GPT-2 retaining 95% accuracy); H&F Box 2 requires cross-validation across methods.*

***Scoping, stated in the discussion***: causal experiments establish that a latent is **load-bearing within ESM-2's own computation** — not that it corresponds to a causal mechanism in protein biology. Conflating these would be the exact overreach H&F warns against.

### M21. C1 — Ablation with a site-specific readout

**The design problem**: InterPLM's readout was P(Glycine), because glycine is a *token* ESM-2 predicts. **ESM-2 has no PTM logit.** A readout must be constructed.

**Protocol**:
1. Forward to layer L; decompose into SAE reconstruction + error term.
2. Zero the candidate latent in the reconstruction; recombine with the **unmodified error term**; splice back; continue.
3. Measure **KL divergence between original and ablated MLM output distributions**, separately at **PTM-site positions** and at **matched control positions** (same residue identity, same stratum, unannotated).
4. Report the ratio KL(sites)/KL(controls).

**Interpretation**: ratio ≈ 1 → the latent contributes uniformly wherever it fires and is not doing PTM-specific work, even if its activation correlates. Ratio ≫ 1 → load-bearing specifically at PTM sites.

*Why KL rather than VF's Δlogit*: VF measured effect on a **probe built afterwards**. KL on the MLM output measures effect on ESM-2's own pretraining-objective behaviour — no auxiliary trained component, no arbitrary probe architecture. It is also the full-distribution version of the loss-recovered idea already in M5, reusing existing machinery. *VF's probe-Δlogit variant is also computed, for numerical comparability with VF Table 8.*

**Implementation**: PyTorch forward hooks on the target layer. InterPLM used NNsight; plain hooks suffice and avoid a dependency.

**Cost**: ~1–2 h.

---

### M22. C2 — Steering with a sequon-completion readout

**The insight**: N-glycosylation is defined by the **N-X-S/T sequon** (X ≠ proline) — composed of exactly the tokens ESM-2 predicts. So InterPLM's periodic-glycine propagation experiment has a direct, non-contrived PTM analogue.

**Protocol**:
1. Take a sequence with an asparagine at position *i* where the latent activates strongly.
2. **Mask position *i*+2** — the sequon's S/T slot.
3. Clamp the latent at position *i* to {0×, 0.5×, 1×, 2×, 4×} of its maximum observed activation (InterProt's range).
4. Measure **P(Ser) + P(Thr)** at the masked position *i*+2 vs steering strength.

**The prediction**: if the latent encodes "this asparagine sits in a glycosylation sequon," amplifying it should raise the model's belief that *i*+2 is S or T — at a **different, unmodified position**. That is InterPLM's propagation criterion.

**Controls — all three necessary:**

| Control | Purpose |
|---|---|
| Random latents (n = 20) | Baseline effect of perturbing anything |
| **Asparagine-identity latents** (V2, ≥7/10) | InterPLM's contrast: their glycine-specific latents (F1 .995, .990, .86) shifted only the steered position and did **not** propagate |
| **Non-SAE direction** — raw difference-of-means vector between glycosylated and non-glycosylated N in the residual stream | **AxBench: simple baselines can outperform SAE steering.** Without this, success does not establish that the *SAE decomposition* is doing the work. |

> **⚠ POST-IMPLEMENTATION CORRECTION — CF-7: the self-gain fix is right but incomplete.**
> Changelog entry #19 correctly identified that the original steering math assumed encoder and decoder
> rows are dual (dot product 1), which holds for a tied autoencoder but not for an independently-learned
> L1 or TopK SAE, and correctly re-derived the injection coefficient as
> $c = (\text{target} - \text{preact})/\alpha$ with $\alpha = d_i \cdot e_i$, reading the true
> pre-activation rather than the post-ReLU value. **The math is right.** Three things it does not yet
> handle, each of which can silently corrupt a dose-response curve:
>
> **(i) Small or negative self-gain.** If $\alpha \approx 0$, $c$ explodes and the injection swamps the
> residual stream; if $\alpha < 0$, steering moves the latent the wrong way. Add the
> `steering_min_self_gain` guard (§0.2, default 0.05): exclude latents with $|\alpha|$ below it, and
> **report how many candidates were excluded per arm** — a high exclusion rate is itself a finding about
> that SAE's geometry.
>
> **(ii) Collateral activation shift.** Adding $c \cdot d_i$ changes *other* latents' activations too,
> since $d_i$ is not orthogonal to other encoder rows. This is a real property of the intervention, not
> a bug — but it must be **measured and reported**: the L2 norm of the change in all other latents'
> activations at the steered position, relative to the intended change in the target. A steering effect
> accompanied by large collateral shift is a weaker claim than one that is clean.
>
> **(iii) TopK discontinuity.** In Arm A, changing the pre-activation can change *which* latents make the
> top-$k$ cut, so the response to steering strength may be piecewise rather than smooth. Record top-$k$
> membership before and after at each steering level, and flag any dose-response curve where membership
> changed — an apparent non-linearity may be a membership switch, not a saturation effect. This does not
> arise in Arms B and C (L1, no hard cut-off), which makes it a genuine A-vs-B reporting asymmetry to
> state rather than smooth over.

**Free additional readouts**: P(Pro) at *i*+1 should **decrease** (proline at X blocks glycosylation) — a sharper asymmetry test than S/T alone; and a position-wise profile *i*−5 to *i*+5, giving InterPLM's Fig 7 layout.

*Why N-glycosylation only*: O-GlcNAcylation and succinylation have no strict consensus, so there is no token-level signature to read out. Forcing it there would be bad design; C1 and C3 carry the causal argument for those types.

**Cost**: ~1 h.

---

### M23. C3 — In-silico mutagenesis (sequence → latent)

**Not present in any paper in the reading list**, and the cheapest causal experiment available — no gradients, no probe, just forward passes. **It runs the opposite direction** from all published steering protocols.

**Loss-of-function**: take a sequon N-X-S where the latent fires. Mutate **S → A**, destroying the sequon while changing one residue. Measure the activation change at the **unmutated asparagine**.

**Gain-of-function**: take an asparagine not in a sequon where the latent is silent. Mutate *i*+2 → **S**, creating a sequon. Measure activation change at the asparagine.

**Specificity controls:**

| Mutation | Expected if the latent reads the sequon |
|---|---|
| S → T at *i*+2 | **No drop** (T also completes a sequon) |
| S → A at *i*+2 | **Large drop** |
| X → P at *i*+1 | **Large drop** (proline blocks glycosylation) |
| Distal at *i*+10 | **No drop** |

**The S→T control is the discriminating one.** A latent merely detecting "serine two positions downstream" drops on S→T. A latent that has learned the sequon *rule* does not. This single comparison separates pattern-matching from rule-encoding.

**Extension to other types**: no consensus motif exists for O-GlcNAcylation or succinylation, but the residue itself can be mutated — **K → R** for succinylation preserves positive charge and approximate size while removing the ε-amino group that is the actual modification target. A latent encoding "modifiable lysine" should drop; one encoding "positively charged residue in this context" should not.

> **A confound specific to the K→R variant, absent from the sequon variant.** In the sequon experiment the measured residue (the asparagine) is **not** the mutated one, so the readout is clean. In K→R the measured residue **is** mutated, so *any* latent with a lysine-identity component drops trivially — including latents that have nothing to do with succinylation. The raw drop is therefore uninterpretable.
>
> **Required control**: report the K→R drop **relative to the drop shown by activity-matched non-PTM lysine-firing latents** (already stored in M15 for M24's Tier-2 baselines). The differential — how much more the candidate drops than a generic lysine-context latent — is the interpretable quantity. Without it, this experiment measures amino-acid identity, not modifiability.

*Why this is strong evidence*: it establishes the latent responds to the **biological determinant** rather than correlated context, which no correlational analysis can claim. Together with C2 it gives a two-way argument: the latent influences the model's sequon predictions, **and** the sequon controls the latent's activation.

**Cost**: ~30 min. **Highest evidential return per unit compute in the whole plan — if time is short, run C3 first.**

---

> **⚠ IMPLEMENTATION-GAP CHECK — CF-8: C3's K→R differential control.** The changelog contains no C3
> entry beyond the existence of `14_m23_m24_mutagenesis_assembly.py`. Verify that the K→R variant reports
> the candidate's activation drop **relative to the drop shown by activity-matched non-PTM lysine-firing
> latents**, not the raw drop. In the sequon variant (S→A) the measured residue is not the mutated one,
> so the readout is clean; in K→R the measured residue *is* mutated, so any latent with a lysine-identity
> component drops trivially. Without the differential, this experiment measures amino-acid identity rather
> than modifiability, and would produce a large, meaningless effect for every lysine-context latent.

### M24. Causal baselines and assembly

**Computation**: every causal effect as a z-score against an empirical control distribution:
$$z = \frac{x_{\text{observed}} - \mu_{\text{controls}}}{\sigma_{\text{controls}}}$$

**Three control tiers, not one:**

| Tier | Controls | Rules out |
|---|---|---|
| 1 | 20 random latents | Generic perturbation effects |
| 2 | Latents **matched on firing rate and mean activation** but with no PTM enrichment | That the effect is driven by how *active* the latent is rather than what it encodes |
| 3 | **Non-SAE directions** — difference-of-means; random unit vector in the residual stream | **AxBench**: that the SAE decomposition is doing the work at all |

*Tier 2 is the one most often omitted in this literature and matters here*: a highly active latent perturbs the model more than a sparse one regardless of meaning.

*Reporting*: VF's threshold (both ablation and steering z ≥ 1 SD) reported for comparability; the **primary criterion is the M8 permutation-null percentile**, consistently with every other threshold.

**Emits**: **F8** — (a) C1 KL-ratio z-scores against all three tiers; (b) C2 dose-response with candidate / N-identity / random / non-SAE lines; (c) C2 position-wise profile; (d) C3 mutagenesis bars with S→T marked as the discriminating control.

**Interpretation matrix — all four cells reported:**

| | **Causally load-bearing** | **Not causally load-bearing** |
|---|---|---|
| **Statistically enriched** | Strongest claim: encodes the PTM and ESM-2 uses it | **VF's case.** Tracks the PTM but the model does not rely on it — redundant, or downstream of the real mechanism. Reportable and interesting. |
| **Not enriched** | Rare and interesting: causally involved without site-level correlation; suggests a distributed or indirect role | Uninformative |

*VF's central lesson is that the top-right cell is **common** — their best-discriminating node lived there. Reporting only the top-left would reproduce exactly the selective-reporting pattern this pipeline exists to avoid.*

**Total Phase VI cost**: 5–7 h GPU including controls.

---

## Phase VII — Assembly

> **⚠ POST-IMPLEMENTATION NOTE — CF-9: the integration fixture's label density is not realistic.**
> Changelog entry #20 raised the fixture from 100% to ~60% K/N residue annotation so that
> `matched_control_positions()` has genuine unannotated background to draw on. That fix was correct and
> necessary. But **60% is still roughly two orders of magnitude denser than reality** — succinylation
> covers ~1% of lysines, N-glycosylation a similar fraction of asparagines. Consequences to state
> wherever integration-test numbers appear: enrichment values, AUPRC (whose baseline *is* the positive
> rate), KL ratios and S9 Jaccards from the fixture are **not** indicative of what the real corpus will
> produce, and in particular AUPRC will look far better on the fixture than it can on real data.
>
> The fixture's job is to prove the chain runs end-to-end and that invariants hold, not to preview
> results. **Add a second fixture variant at ~1–2% density** as a power-realism check — it will reveal
> whether any module degrades or divides by zero when positives are genuinely rare, which is the regime
> every real PTM type except phosphorylation actually occupies.

### M25. Ensemble analysis, final figures, release

**Ensemble** (adopting VF's proposed resolution to "no single node passes everything"): for each PTM type, report the set of G1+G2 passers characterised jointly —
- how many latents, which arms, which layers
- **decoder-direction similarity** — one direction or several? (near-parallel decoders = one concept split across latents; orthogonal = distinct sub-mechanisms)
- **co-firing structure** — same sites, or partitioned?
- **joint k-sparse probing** vs the best single latent (V11)

*Why this is the right unit of claim*: feature splitting means a concept may genuinely not live in one latent. Connects to InterPLM's clustering finding — their TBDR beta-barrel cluster held one true specialist (F1 = .998) plus two broader detectors (.793, .611) sharing a decoder neighbourhood. The PTM analogue is directly checkable here.

**Final figures**: **F1** (study design schematic — three arms, 21 combinations, three-pass architecture, stratified design, with panel (d) showing the naive-vs-stratified contrast conceptually); **S2** (UMAP of candidate decoder vectors with within-cluster cross-activation).

**Release artefacts:**

| Artefact | Contents |
|---|---|
| Code | Three-pass pipeline; permutation-null module; validation cascade; causal hooks |
| Processed datasets | Filtered site tables per type per stratum with evidence tier, LP class, source count, crosstalk flag, **discovery/holdout split label**; the exclusion mask; **`split_manifest.json`** (cluster assignments and seed, so the split is exactly reproducible); the Kaggle Dataset build script |
| **Histograms** | Per-(latent, label) 64-bin arrays, all 21 combinations × {stratified, naive} × {discovery, holdout} (~2 GB) — **the key reusable artefact** |
| Full results | T4 and T6 complete, not only the reported subset |
| Null distributions | Permutation results, so thresholds are auditable |

*Releasing the histograms is unusual and deliberate*: it lets any reader recompute any statistic at any threshold without re-running inference — a stronger reproducibility guarantee than code alone, and it makes the τ-tuning objection permanently unanswerable.

---

# PART 2 — REFERENCE

## 2.1 Module dependency graph

```
M1 (acquire) ──┬─> M2 (corpus) ──> M3 (labels) ──┐
               └─> M4 (load+calibrate) ──> M5 (fidelity) ═[GATE Tier 0]═┐
                                                                        │
M3 + M4 ──────────────────────> M6 (Pass 1: histograms) <───────────────┘
                                        │
        ┌───────────────────────────────┼─────────────────────────────┐
        v                               v                             v
   M7 (stats)                      M10 (τ sweep)                M14 (seq-level)
        │                               │
        v                               v
   M8 (null) ──> M9 (correct) ──> M11 (layers) ──> M12 (figs) ──> M13 (naive vs strat)
                      │
                      v
                 M15 (Pass 2) ──┬─> M16 (V1–V6)
                                ├─> M17 (V7–V9)   [needs AlphaFold]
                                ├─> M18 (V10–V11) [needs raw-activation probe]
                                └─> M19 (V12)     [needs OOD inference]
                                          │
                                          v
                                   M20 (profile) ──> M21,M22,M23 ──> M24 ──> M25
```

## 2.1b Post-implementation correction index

Corrections raised by auditing `IMPLEMENTATION_CHANGELOG.md` against this plan. Each is written inline at
its module; this table is the checklist.

| ID | Module | Severity | Correction |
|---|---|---|---|
| **CF-1** | M5 | Diagnostic | Report normalisation overflow (clamp) rate in T3; raise `calibration_n_proteins` if >1% |
| **CF-2** | M9 | **Scope error** | Pass 2 and Tiers 1–4 take **stratified-background candidates only**; naive results feed M13 alone |
| **CF-3** | M7 | **Verify exists** | `effective_types_claimed`; chemically-impossible fraction; naive-background Bonferroni *m* |
| **CF-4** | M14 | **Wrong test + invalid comparison** | Step 4 → one-vs-rest Mann-Whitney; S9 universe disclosed, control-differenced, and **not** compared to COMPASS-PTM's 0.6430 |
| **CF-5** | M15–M24 | **Stale plan reference** | `anchor_layer_set()` must iterate all arms; Arm C anchors `[4, 6]`; update entry #25's docstring |
| **CF-6** | M16 | **Borrowed threshold** | G4 = Bonferroni-significant + above 95th null percentile vs Hard negatives; stop reusing `v6_crosstalk_retention_min` |
| **CF-7** | M22 | **Incomplete fix** | Self-gain guard; report collateral shift; flag TopK top-*k* membership changes |
| **CF-8** | M23 | **Verify exists** | K→R reported as a differential against activity-matched lysine-firing controls |
| **CF-9** | Integration | Fixture realism | Add a ~1–2% density fixture variant; do not read fixture numbers as indicative |
| **CF-10** | **M7 / §0.3** | **CRITICAL** | Label domain $\ell$ for selectivity, enrichment and effective-types is **PTM types only** — including "unmodified" silently flattens all three metrics to constants |
| **CF-11** | M8 | **Blocks CF-6** | Add a negative-tier dimension to the permutation null so G4's "above 95th null percentile vs Hard negatives" is computable |
| **CF-12** | M14 / §0.2 | **Filters everything out** | SPIRAL's 0.05 sequence-mean floor assumes their L₀ ≈ 12.6% density; Arm A is ~8× sparser. Keep the coverage clause, drop the magnitude clause |
| **CF-13** | M9 / M16 | **CRITICAL** | Residue-dominance is **tautological** under stratification and would hard-exclude every true positive. Gate applies to the **naive background only**; replaced for stratified candidates by a reported target-residue-preference diagnostic |
| **CF-14** | M19 | **Wrong polarity** | VF's Stage 6 rewards *low* OOD firing; this check needs *replication*. Drop activation ratio; re-run the three-part decomposition on the OOD corpus against an OOD-specific null |
| **CF-15** | M18 | Precondition | Report the raw-activation probe's own AUPRC vs baseline before any absorption result; absorption is uninterpretable when the probe is near chance |
| **CF-16** | Corpus construction | Will error or mis-stratify | Split is multi-label; use iterative stratification, report achieved proportions |
| **CF-17** | Label construction | Silent false negatives | Masked residues must leave the tally entirely; $g_\ell$ is per (stratum, type) |
| **CF-18** | Calibration | Divide-by-zero | Dead latents: scale 1.0, mark, exclude from tallies |
| **CF-19** | Statistical testing | **Common, not rare** | Zero modified-residue firings ⇒ three effect sizes undefined; count as a category, don't NaN-propagate |
| **CF-20** | Statistical testing | Uncomparable p-values | One-sided Fisher vs two-sided χ²; make direction consistent |
| **CF-21** | Specificity validation | Undefined for a primary target | Single-type strata ⇒ G3 `not_applicable`; asparagine likely affected |
| **CF-22** | Generalisation | Meaningless tests | Build a species × type applicability matrix; *E. coli* lacks N-glycosylation |
| **CF-23** | Layer analysis | Selection circularity | Choose the peak layer on the discovery partition |
| **CF-24** | Correction / null | Category conflation | Permutation null gives thresholds and FDR, never per-latent p-values |
| **CF-27** | Permutation null | **CRITICAL — unimplementable as written** | The shuffle mechanism was never specified. Use multivariate-hypergeometric resampling from the stored histograms; decile-binned nulls for screening, exact nulls for candidates; six edge cases enumerated |
| **CF-25** | Fidelity gate | **Indefensible exclusion** | Only InterPLM-8M has published fidelity. Replace the comparative gate with a broken-instrument floor plus mandatory reporting; let the reader judge |
| **CF-26** | Acquisition / labels / streaming / generalisation | **New requirement** | Cross-database replication via qPTM is now adopted, not optional. Must be built **before** the streaming pass |

**Priority order for the implementor.** CF-10 and CF-3 both invalidate the whole pipeline, so apply them
together first, before spending compute downstream:

1. **CF-10 + CF-3** (M7 — full rebuild follows)
2. **CF-13** (M9 — currently hard-excludes every true positive; nothing downstream is meaningful until fixed)
3. **CF-11** (M8 — required before CF-6 can run)
4. **CF-2 + CF-5** (change what data Phase V/VI produce)
5. **CF-6, CF-4a, CF-12, CF-14** (wrong or unusable statistics)
6. **CF-7, CF-8, CF-15** (causal and probe correctness)
7. **CF-1, CF-4b, CF-9** (diagnostics, disclosure, fixture)
8. **CF-16…CF-24** (degenerate cases and statistical validity — Part 0b). Apply **CF-19 together with
   CF-10**, since it is a direct consequence of that change and will otherwise surface as widespread NaN

**The three that would silently produce a null result** — CF-10 (all effect sizes flatten to constants),
CF-13 (every true positive hard-excluded), CF-12 (sequence-level track filtered to nothing) — share one
root cause: a formula or threshold transplanted from a source domain where the underlying assumption held,
into PTM data where it does not. Any future borrowed constant or criterion should be checked against the
question *"what does this assume about the label distribution, and is that true for PTM sites?"* before use.

See §2.1c for cache invalidation, which must accompany every one of these.

## 2.1c Applying the corrections — cache invalidation and fingerprint fields

**This section exists because the notebook has a config-fingerprint caching layer, and changelog entries
#4 and #12 both record the same failure: a config value changed, but the cache was not invalidated, so a
stale result was served with no error.** Applying CF-1…CF-11 without addressing this will reproduce that
bug at scale — the code will be corrected and the outputs will not.

**Step 1 — register the new config keys in the fingerprint field list.** `profile_sample_n_residues`,
`steering_min_self_gain`, `v5_hard_tier_criterion`. A key absent from the fingerprint list is a key whose
change is a silent no-op (entry #4's exact failure mode). Also add the fields entry #12 found missing if
they are still absent: `min_firings_residue_level`, `tau_sweep`, `expected_count_min`.

**Step 2 — force invalidation of everything downstream of each correction**, not just the corrected module:

| Correction | Must rebuild |
|---|---|
| CF-10 (label domain) | **M7 → everything after it.** This changes three effect sizes, so T4, M8's nulls, M9's candidate ranking, and every table and figure downstream are all affected |
| CF-11 (tier nulls) | M8, M16 (V5/G4), T6, F7 |
| CF-2 (candidate scope) | M9 → M15 → M16–M25. M13 is **not** affected (see below) |
| CF-3 (missing M7 outputs) | M7 → everything after it |
| CF-5 (Arm C) | M15 → M16–M25 |
| CF-6 (G4 criterion) | M16, T6, F7 |
| CF-4a (M14 test) | M14 / S8 only |
| CF-4b (S9 universe) | S9 only |
| CF-7, CF-8 (causal) | M21–M24, F8 |
| CF-1 (overflow diagnostic) | M5 / T3 only — unless the clamp rate exceeds 1%, in which case M4 re-runs and **everything** rebuilds |

**Step 3 — two ordering hazards.** CF-10 and CF-3 both touch M7 and therefore invalidate the entire
pipeline; apply them **together, first**, before spending compute on anything downstream. CF-1 can also
trigger a full rebuild, so run it early enough to find out.

**Two clarifications that prevent CF-2 from breaking something that currently works:**

- **M13 does not read `candidates.json`.** T5 and F4 are built from M7/M9 *statistics tables* (passer counts
  per background) plus M6's amino-acid firing matrix (residue-dominance per naive passer). Removing naive
  candidates from the candidate list must not remove naive rows from the results tables. If the current
  implementation has M13 reading the candidate list, that coupling must be broken as part of CF-2, or the
  headline methodological result disappears.
- **Entry #16's `background`-aware lookup should be kept** as defensive code even though CF-2 makes it
  unreachable. It costs nothing and guards against the coupling reappearing.

**One clarification on CF-4a**: the protein-level one-vs-rest negative set should be restricted to proteins
containing **≥1 residue of the relevant stratum**. A protein with no lysines cannot be succinylated, and
including it is the protein-level analogue of the residue-identity confound §0.4 removes at residue level.
The effect is small (most proteins contain lysines) but the restriction is free and keeps the two levels
methodologically consistent.

## 2.2 Emission index — what is produced where

| Artefact | Module | Type |
|---|---|---|
| T1 dataset provenance | M3 | table |
| T2 label composition | M3 | table |
| T3 instrument validation | M5 | table |
| T4 core statistical results | M9 | table |
| T5 naive vs stratified | M13 | table |
| T6 validation profile | M20 | table |
| F1 study design | M25 | figure |
| F2 dataset composition | M3 | figure |
| F3 instrument validation + ID | M5 | figure |
| F4 naive vs stratified | M13 | figure |
| F5a–b layer rate & strength | M11 | figure |
| F5c–f core statistics | M12 | figure |
| F6a architecture A vs B | M11 | figure |
| F6b τ-stability | M10 | figure |
| F6c–d mechanism & motif | M17 | figure |
| F7 validation profile | M20 | figure |
| F8 causal | M24 | figure |
| F9 cross-species | M19 | figure |
| S1 activation landscape | M12 | supp |
| S2 decoder UMAP | M25 | supp |
| S3 per-family AUPRC | M16 | supp |
| S4 full τ grid | M10 | supp |
| S5 kNN baselines | M19 | supp |
| S6 structure renderings | M17 | supp |
| S7 accessibility | M17 | supp |
| S8 sequence-level track (one-vs-rest Mann-Whitney, Cliff's δ, length-confound diagnostic) | M14 | supp |
| S9 protein-level event coherence | M14 (after M15) | supp |

## 2.3 Statistical toolkit — where each test is used

| Toolkit ID | Test | Module | Role |
|---|---|---|---|
| B1 | Chi-square | M7 | Occurrence, expected cells ≥ 5 |
| B2 | Fisher's exact / hypergeometric | M7 | Occurrence, rare types |
| B3 | Bonferroni | M9 | **Primary** correction |
| B4 | Benjamini–Hochberg | M9 | Secondary |
| B5 | Selectivity | M7 | Occurrence effect size |
| — | **Effective types claimed** (exp. Shannon entropy) | M7 | Occurrence effect size — severity companion to selectivity |
| C2 | Fold enrichment | M7 | Occurrence effect size |
| C3 | Odds ratio | M7 | Companion to Fisher |
| — | **Chemically-impossible-association fraction** | M7 (naive background), M13 | Chemistry-grounded false-positive counter |
| — | **Protein-level event coherence** (mean per-protein Jaccard) | M14 step 5 | Program-level coherence above the residue |
| A2 | Mann-Whitney U | M7 | Magnitude given firing |
| — | Cliff's δ | M7 | Magnitude effect size |
| G2 | AUPRC | M7 | **Primary** discrimination |
| D1 | AUROC | M7 | Secondary, VF comparability |
| A1/C1 | Welch's t / Cohen's d | M7 | VF numerical comparability only |
| A2 | Mann-Whitney U + Cliff's δ (one-vs-rest) | M14 step 4 | **Sequence-level primary** — handles protein-level multi-label overlap |
| A3/A4 | Kruskal-Wallis + η², rank-residualisation | M14 | Rank-residualisation always; Kruskal-Wallis **optional secondary**, singly-modified proteins only |
| E1 | Spearman ρ | M17 | Activation vs solvent accessibility |
| F1 | Z-score vs empirical controls | M24 | All causal effects |
| G5 | Balanced accuracy | M19 | kNN utility probe |
| H1–H4 | EV, cosine, KL, % Loss Recovered | M5 | Fidelity gate |
| I1 | TwoNN intrinsic dimension | M5 | Layer characterisation |
| J1/J2 | Feature absorption, k-sparse probing | M18 | Representational integrity |
| — | Permutation null (1,000) | M8 | **All thresholds**; third correction |

**Unused, with reason**: E2 graph-distance monosemanticity (needs a PTM DAG — named extension); G1 MCC, G3 F_max/S_min, G4 R²/MAE (downstream-classifier metrics; no classifier is the deliverable); J4 SCR/TPP (subsumed by the stratified design, which removes the residue confound by construction rather than by ablation).

## 2.4 Compute budget

| Phase | Modules | GPU | Wall-clock |
|---|---|---|---|
| I Data | M1–M3 | none | 2–4 h |
| II Instrument | M4–M5 | yes | ~40 min |
| III Measurement | M6–M9 | M6 only | 1–2 h GPU + ~1 h CPU |
| IV Histogram analysis | M10–M14 | none | 1–2 h |
| V Targeted | M15–M20 | M15, M19 | ~2 h GPU + 2–3 h CPU (Arm C adds ~5 min: 8M forward passes are ~80× cheaper) |
| VI Causal | M21–M24 | yes | 5–7 h for Arms A+B; Arm C adds <1 h |
| VII Assembly | M25 | none | 1–2 h |

**Total GPU: ~10–12 h**, within Kaggle's 30 GPU-h weekly allocation. **VRAM: ~3 GB against 16 GB/T4.** **Disk**: ~6 GB models+data, ~2 GB histograms (4 variants per combination), ~1.2 GB targeted activations + profile samples (9 anchor combinations) (`/kaggle/temp`). Every storage figure is well inside Kaggle's 20 GB `/kaggle/working` and ~73 GB scratch.

## 2.5 Minimum viable path (Monday supervisor deliverable)

| Priority | Modules | GPU | Yields |
|---|---|---|---|
| 1 | Part 0 (this document) | none | The design itself |
| 2 | M1–M3 | none | **T1, T2, F2** |
| 3 | M4–M5 | ~40 min | **T3, F3** |
| 4 | M6–M9, M13 — *one PTM type, one layer* | ~30 min | **F4** (the headline methodological figure) |

**Recommended framing**: lead with the novelty ledger (the unoccupied cell across four research groups and three model families), then the residue-stratification argument with its worked 17× arithmetic, then whatever data exists. **The methodological argument is the contribution; the numbers substantiate it.** If no GPU time is available before Monday, priorities 1–2 alone are defensible — the design is the deliverable at this stage, and T1/T2 show the dataset work is real rather than planned.

## 2.6 Novelty ledger — the unoccupied cell

| Paper | Level | Concept vocabulary | Statistical rigour | PTM-specific? |
|---|---|---|---|---|
| InterPLM | residue | 433 Swiss-Prot concepts **incl. `ft_mod_res`, `ft_carbohyd`** | domain-adjusted F1 only; no p-value, no correction | PTM-adjacent, **aggregate not type-specific**, Swiss-Prot not dbPTM |
| InterProt | residue + protein | InterPro families | F1 > 0.7, swept threshold, no correction | no |
| VG&A | residue | 12 UniProt categories **incl. "Glycosylation site"** | precision OR recall > 0.80, **no p-values at all** | one undifferentiated glycosylation category |
| Gujral | protein | GO terms | **hypergeometric + BH** — genuinely rigorous | not PTM; residue-level arm drops to LLM captioning |
| ESMC-SAE | protein | EC classes | mutual information, **no correction** | 698 PTM features exist but are GPT-5-labelled, unvalidated |
| VF | protein | β-lactamase classes | Welch + Cohen's d, **no correction**; 6-stage validation on CLT only | no |
| SPIRAL | nucleotide + sequence | 7 structure classes, 16 RNA types | **χ² + Bonferroni + selectivity + enrichment + η² + rank-residualisation** | RNA, not protein |
| ProtSAE | residue | GO terms | relevance-F1; concepts supervised in by construction | deubiquitination only as a steering target |
| ProtSyntax / PTMGPT2 | residue | 40 PTM classes | supervised prediction metrics | yes — but **no unsupervised representation decomposition at all** |
| **COMPASS-PTM** (*Nat. Commun.* 2026) | residue + protein + enzyme | multi-label across dbPTM-ML / qPTM-ML / PTMint-MC | macro F1, MCC, spurious multi-label burden, protein-level event-F1, integrated-gradient attribution | **yes — the strongest PTM work to date, and fully supervised.** Its interpretability is *post-hoc attribution on a trained predictor* plus UMAP of its own learned embeddings. It never asks what an *unsupervised* representation already encodes, and its crosstalk prior is an architectural input rather than a tested hypothesis. |

**The unoccupied cell**: residue-level × formally corrected significance + selectivity + enrichment × wet-lab-curated PTM-type-specific databases × unsupervised features. Confirmed unoccupied across three model families (ESM-2, ESMC, BiRNA-BERT) and four independent research groups.

**Sharpest citable admissions**:
- ESMC-SAE §3.5: GPT-5 annotations "may contain errors or oversimplifications; expert curation… would strengthen interpretability claims."
- InterPLM Fig 10: Swiss-Prot F1 vs LLM-caption accuracy **r = 0.11** — the two validation modes measure different things.
- H&F Box 2: "wet-lab validation of XAI-derived insights has not been explored to date."
- Gujral §5: SAEs/transcoders "stand at a significant disadvantage to even the simplest of supervised methods" for prediction.
- VF: top AUROC = 1.000 node **fails** causal validation.

## 2.7 What this pipeline contributes

| Contribution | Prior state |
|---|---|
| Occurrence/magnitude decomposition (M7, F5e) | No paper separates them |
| **Severity-aware selectivity — effective types claimed** (§0.3, M7) | COMPASS-PTM separates frequency from severity for a *supervised predictor's* spurious labels; no SAE paper does the equivalent for a latent's label spread |
| **Chemically-impossible-association counter** (M7 step 7, M13) | COMPASS-PTM observes the failure (lysine-only PTMs predicted on threonine) in a baseline predictor; nobody uses it as a quantified false-positive counter |
| **Protein-level event coherence** (M14) | COMPASS-PTM introduces protein-level event-F1 for predictions; never applied to latent firing patterns |
| Residue-stratified backgrounds with the naive comparison run deliberately (M13, T5, F4) | No paper quantifies the residue-identity confound |
| Three-tier negative hierarchy (M3, V5) | Every prior negative set is "no annotation exists" |
| Ambiguity masking incl. LP-tiering (M3 Step 3) | No paper masks ambiguous sites; none uses LP as a confidence tier |
| Crosstalk sensitivity analysis (V6) | No paper acknowledges PTM co-occurrence contaminating one-vs-rest tests |
| Empirically calibrated thresholds from a permutation null (M8) | VF imported fixed bands from a much easier task |
| Validation **profile** rather than verdict (M20) | VF collapsed to Pass/Mixed/Fail |
| Six-way specificity validation of **SAE latents** (M16) | VF applied its stages to a CLT circuit, not SAE features |
| Structural-vs-sequential mechanism split for PTMs (V7, F6c) | InterPLM built the method; nobody applied it to PTMs |
| Site-specific KL ablation readout (M21) | VF used probe-Δlogit; no PTM-site-specific causal readout exists |
| Sequon-completion steering (M22) | InterPLM's propagation design, never applied to a PTM motif |
| **In-silico mutagenesis, sequence → latent (M23)** | Absent from the entire reading list; the reverse causal direction |
| Non-SAE control directions (M24) | AxBench's warning unaddressed in every protein SAE paper read |
| Pre-registered contingency table (§0.8) | Absent from every paper read |
| Released histograms as a reusable artefact (M25) | No SAE-on-pLM paper releases anything permitting recomputation at arbitrary thresholds |
