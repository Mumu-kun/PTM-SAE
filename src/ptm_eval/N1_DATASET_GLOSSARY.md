# N1 Dataset Glossary

**Status: FINAL — every number measured from the completed N1 run.** Nothing here is an estimate.

**Provenance: re-verified end-to-end against the local-machine run through CL#57** (the run whose
saved outputs are in `PTM_N1_Corpus_Labels_local_outputs.ipynb`; M2 and M3 both built fresh, no
cache). Every figure below was re-read from that run's printed output.

What changed against the previously-recorded run, and what did not. **Unchanged:** corpus size,
residue counts, the ten stratum totals, cluster count, every per-type total site count, every
positive rate, every crosstalk fraction, all label/mask/qPTM/OOD row counts, the gold-negative
figures, T5's full and strict testability, and every cross-source overlap number. **Changed:** the
discovery/holdout **split assignment** — 14,384/3,744 at 20.65% rather than 14,380/3,748 at
20.68%, with 9,110 singletons rather than 9,106, and correspondingly different per-type discovery
and holdout counts throughout §2. The corpus and the labels are identical; only which side of the
split each cluster landed on moved. **Anything downstream keyed to the split — N2's
discovery/holdout histograms, V12b's held-out re-estimation — must be built against the
`corpus.parquet` from this run, not an earlier one.**

Scope: dataset facts only — what each source contains, what survived, and what the resulting
label sets look like. Design reasoning lives in `N1_CHANGELOG.md`; code logic in
`N1_CODE_WALKTHROUGH.md`.

---

## 1. Headline outcome

| | Measured |
|---|---|
| Corpus | **18,128 proteins**, 7,474,423 residues |
| Clusters (CD-HIT 40%) | **11,909** |
| Split | discovery 14,384 / holdout 3,744 — **20.65% holdout** |
| Types entering the cascade | **47** |
| Types clearing the 200-site floor | **21** |
| **Types clearing the 1,000-site headline bar** | **20** |
| Total labelled sites | **498,619** |
| qPTM replication sites | 131,700 |
| OOD sites (4 species) | 357,713 full / 139,310 strict (no human homolog) |
| Output dataset size | **52.7 MB** |

All four gates pass. The plan's contingency (reduce `holdout_fraction` to 0.15) is not triggered.

---

## 2. The 20 headline-eligible types

Every one is `v12b_eligible` — ≥200 sites in **both** discovery and holdout.

| # | Type | Stratum | Sites | Discovery | Holdout | Positive rate | **Crosstalk** |
|---|---|---|---|---|---|---|---|
| 1 | phosphorylation_ST | ST | 228,365 | 179,077 | 49,288 | 23.22% | 2.0% |
| 2 | ubiquitination | K | 89,348 | 69,780 | 19,568 | 21.02% | 43.4% |
| 3 | acetylation | K | 38,225 | 29,717 | 8,508 | 8.99% | 72.3% |
| 4 | **phosphorylation_Y** † | Y | 35,903 | 27,998 | 7,905 | 17.15% | **0.06%** |
| 5 | sumoylation | K | 32,237 | 25,251 | 6,986 | 7.58% | 61.7% |
| 6 | **N-glycosylation** ★ | N | 13,981 | 11,023 | 2,958 | 5.31% | **0.04%** |
| 7 | crotonylation | K | 9,571 | 7,322 | 2,249 | 2.25% | 93.3% |
| 8 | 2-hydroxyisobutyrylation | K | 7,148 | 5,465 | 1,683 | 1.68% | 95.3% |
| 9 | O-glycosylation | ST | 6,605 | 5,091 | 1,514 | 0.67% | 58.9% |
| 10 | **O-GlcNAcylation** ★ | ST | 6,132 | 4,773 | 1,359 | 0.62% | 69.0% |
| 11 | glycation | K | 4,354 | 3,296 | 1,058 | 1.02% | 39.7% |
| 12 | **methylation_R** † | R | 4,156 | 3,147 | 1,009 | 0.96% | **0.22%** |
| 13 | malonylation | K | 4,121 | 3,248 | 873 | 0.97% | 93.1% |
| 14 | **sulfoxidation** † | M | 3,910 | 3,033 | 877 | 2.37% | **0.00%** |
| 15 | methylation_K | K | 3,702 | 2,757 | 945 | 0.87% | 65.5% |
| 16 | **succinylation** ★ | K | 3,627 | 2,778 | 849 | 0.85% | **90.7%** |
| 17 | beta-hydroxybutyrylation | K | 2,290 | 1,750 | 540 | 0.54% | 90.8% |
| 18 | glutathionylation | C | 1,434 | 1,074 | 360 | 0.80% | 26.9% |
| 19 | palmitoylation | C | 1,137 | 886 | 251 | 0.63% | 20.9% |
| 20 | S-nitrosylation | C | 1,035 | 750 | 285 | 0.58% | 34.7% |

★ = headline type · † = exists only because of the residue-aware canonicalisation fix or a new stratum

**21st type:** neddylation (400 sites) clears the 200 floor but not the 1,000 bar.
**23 types pooled** into `<stratum>::other_PTM` *(was 26; corrected 2026-09-15, measured from
`T4_type_survival.pooled_into` and `T1_dataset_provenance.n_types_pooled_to_other`, both 23)*. The
old figure came from 47 − 21 and wrongly counted the **three types with zero surviving sites**
(Iodination, carboxylation, formylation) as pooled; they have nothing to pool. 21 above-floor + 23
pooled = the 44 types in the label file.

---

## 3. Crosstalk — the most consequential dataset property

`crosstalk_fraction` = share of a type's sites that **also carry another PTM type at the same
residue**. 49,495 crosstalk sites overall.

### Clean types (crosstalk < 3%) — isolable

| Type | Crosstalk | Sites |
|---|---|---|
| sulfoxidation | **0.00%** | 3,910 |
| N-glycosylation | **0.04%** | 13,981 |
| phosphorylation_Y | **0.06%** | 35,903 |
| methylation_R | **0.22%** | 4,156 |
| phosphorylation_ST | 2.01% | 228,365 |

### Confounded types (crosstalk > 60%) — hard to isolate

| Type | Crosstalk |
|---|---|
| lactylation | 100.00% |
| glutarylation | 98.91% |
| neddylation | 98.75% |
| 2-hydroxyisobutyrylation | 95.29% |
| carboxymethylation | 95.12% |
| crotonylation | 93.31% |
| malonylation | 93.08% |
| beta-hydroxybutyrylation | 90.79% |
| **succinylation** ★ | **90.65%** |
| phosphoglycerylation | 81.82% |
| acetylation | 72.30% |
| **O-GlcNAcylation** ★ | **68.98%** |
| methylation_K | 65.48% |
| sumoylation | 61.73% |

**What this means.** Two of the three headline types are heavily confounded. 90.65% of
succinylation sites also carry another modification, so "this latent detects succinylation" is
hard to separate from "this latent detects lysine acylation generally". This is real biology —
lysine acylations compete for the same ε-amino group — not a data defect.

**Practical consequence:** the four cleanest types in the set (sulfoxidation, N-glycosylation,
phosphorylation_Y, methylation_R) are the strongest candidates for an unambiguous claim, and
three of those four exist only because of fixes made during this build.

---

## 4. Background populations (*g<sub>ℓ</sub>*)

From `stratum_counts.json`. Exact counts, not sampled.

| Stratum | Residues | Headline types | Status |
|---|---|---|---|
| ST | 983,282 | 3 | active |
| E | 514,269 | 0 | **dead** — best type has 82 sites |
| R | 432,208 | 1 | active |
| K | 425,120 | 10 | active |
| Q | 345,562 | 0 | **dead** — 3 sites |
| N | 263,374 | 1 | active |
| Y | 209,350 | 1 | active |
| C | 179,280 | 3 | active |
| M | 165,084 | 1 | active |
| W | 96,344 | 0 | **dead** — 49 sites |
| *(the ten strata above sum to 3,613,873 residues)* | | | |
| **corpus total, all residues** | **7,474,423** | | |

Of the four strata added this build, only **M (sulfoxidation)** produced a usable type. E, W and Q
are retained for future relabelling but contribute background only. Their pooled `other_PTM`
buckets are themselves below the 200 floor.

> **Dynamic-range caveat.** `positive_rate` caps achievable fold enrichment at `1 / rate`.
> phosphorylation_ST sits at 23.2%, so its ceiling is **4.3×** — a 4× result there is near-perfect,
> while 4× for succinylation (ceiling 117×) would be weak. **Do not compare FE across types
> without accounting for this.**

| Type | Positive rate | Max FE |
|---|---|---|
| phosphorylation_ST | 23.2% | 4.3× |
| ubiquitination | 21.0% | 4.8× |
| phosphorylation_Y | 17.1% | 5.8× |
| N-glycosylation | 5.3% | 18.8× |
| succinylation | 0.85% | 117× |
| O-GlcNAcylation | 0.62% | 160× |

---

## 5. Per-source contribution

| Source | Raw rows | Accessions | Types | Surviving human sites |
|---|---|---|---|---|
| Swiss-Prot | 18,128 | 18,128 | — | *(the corpus)* |
| dbPTM 2025 | 2,228,389 | 280,361 | 37 | largest contributor |
| CPLM 4.0 | 298,634 | 24,327 | 24 | 198,907 after chemistry |
| qPTM | 11,482,553 | 40,730 | **6 only** | 131,700 (separate set) |
| O-GlcNAcAtlas I+II | 46,517 + 14,518 | 7,793 / 3,143 | 1 | 14,884 / 2,805 |
| N-GlycositeAtlas | 38,667 | 7,204 | 1 | 24,829 |
| N-GlyDE | 3,525 | 918 | 1 | 1,088 clean negatives |

**Download total:** 1,679 MB across 54 artefacts.

### Source quirks that matter

| Source | Property |
|---|---|
| **dbPTM** | **Not species-filtered.** Human share varies wildly: Phosphorylation 31.4%, Succinylation **9.3%**, Formylation **0%**. Species only recoverable from the `entry_name` suffix |
| **dbPTM Ubiquitination** | **33% chemistry failure with only 1% out-of-range** — mismatches scatter across L/E/S/A/V in proteome-average proportions. A coordinate problem, not biology. CPLM's identical chemistry passes at 99.3%. ~22,100 sites dropped |
| **dbPTM Acetylation** | 10.2% "failure" is **real N-terminal acetylation** on Met/Ala/Ser — correctly excluded from the K stratum |
| **CPLM** | No header row. `lp` and `flanking_peptide` **100% null** → no LP confidence filter is possible; the mask was removed entirely (CL#50), not left inert |
| **qPTM** | `Organism` reads `Human`, not `Homo sapiens`. Species: Human 53.0% / Mouse 25.3% / Yeast 20.2% / **Rat 1.5%** |
| **O-GlcNAcAtlas** | Not UTF-8 (cp1252 bytes). 29 species, 8 accession registries. `site_residue` contains junk (`'1'`, `'8'`) |
| **N-GlycositeAtlas** | Title row **above** the header row |
| **N-GlyDE** | 99.7% of its positives are already in dbPTM — negatives-only source by design |

---

## 6. Where sites are lost

**Corpus membership is the dominant filter — not chemistry.**

| Source | Total dropped | Not in corpus | Wrong chemistry |
|---|---|---|---|
| CPLM | 99,727 | **91,847 (92%)** | 7,591 |
| dbPTM Ubiquitination | 292,845 | **264,507 (90%)** | 27,521 |
| N-GlycositeAtlas | 13,838 | **13,486 (97%)** | 343 |
| qPTM (human) | 2,017,936 | **1,999,149 (99%)** | 16,906 |

"Not in corpus" = non-human, over 1,022 aa, or not Swiss-Prot reviewed. These are the corpus rules
working as intended.

**The evidence filter drops zero rows from every source** — dbPTM's `/download/experiment/`
endpoint returns only `experimental`, and CPLM/the atlases are set to `MS/MS`. There is nothing
for it to discard.

**Isoform recovery:** 14,868 CPLM rows and 15,765 dbPTM Phosphorylation rows recovered by
stripping `-2` suffixes. All flagged `from_isoform_accession`.

**Deduplication:** 629,098 → 498,619 (verified).

> **`source_count` counts contributing ROWS, not distinct sources.** *(was: "Max `source_count` is
> **2**, structurally — dbPTM ships one file per type, so a K-site can only ever be corroborated by
> CPLM + one dbPTM file"; corrected 2026-09-15 after re-measuring `labels_stratified.parquet`.)*
> `step4_dedup` computes `combined.groupby(["accession","position","type"]).size()`, which counts
> rows. A source that lists the same site under two PMIDs contributes 2. Measured maximum is **52**
> (N-glycosylation), and **ubiquitination reaches 25 although only two sources can supply it** —
> which is the proof the field is not counting sources. CPLM has no internal duplicates (298,634
> rows, 298,634 distinct sites), so the inflation comes from dbPTM and the atlases.
>
> **Consequence for the `min_source_count_high_confidence = 2` robustness subset** (declared in N1's
> config, inherited by N3, first used by N4): it selects **93,446** sites on row multiplicity, not
> 87,100 sites on cross-source agreement. Do not describe it as corroboration until the field is
> either fixed or renamed. See `N1_RESULT_ANALYSIS.md` D1.

| `source_count` | Sites |
|---|---|
| 1 | 405,173 |
| 2 | 80,769 |
| 3–52 (tail) | 12,677 |
| **≥ 2** | **93,446** |

*(was 1 → 411,519 and 2 → 87,100, a two-way partition that matched neither the measured `==1`/`==2`
counts nor `==1`/`>=2`; corrected 2026-09-15, measured from `labels_stratified.parquet`.)*

---

## 7. Negatives and masks

### Three-tier negatives

| Tier | Definition | Available for |
|---|---|---|
| Gold | experimentally verified unmodified | N-glycosylation only |
| Hard | unannotated target residue in a protein carrying ≥1 site of that type (Corpus A) | all types |
| Background | all other target residues (Corpus B) | all types |

**Gold tier was 16.95% contaminated.** 222 of 1,310 N-GlyDE "verified non-glycosylated" sequons
are annotated **positives** in the validated positive set. Removed; **1,088 clean** remain. The
removed set is kept in `gold_negatives_contradicted.parquet`.

### Exclusion mask — 6,382 sites

| `mask_reason` | Sites | Meaning |
|---|---|---|
| `dbptm_o_linked_glyco_may_be_uncatalogued_o_glcnac` | 3,577 | may be O-GlcNAc the Atlas hasn't catalogued |
| `oglcnac_atlas_dataset_ii_ambiguous` | 2,805 | lab couldn't localise the residue |

Masked sites belong to **neither** positives nor negatives. **They must be subtracted from every
background population downstream.**

---

## 8. Cross-source disagreement

Measured site-level overlap:

| Pair | Shared | % of A | % of B |
|---|---|---|---|
| dbPTM N-linked ∩ N-GlycositeAtlas | 7,366 | 26.9% | 50.3% |
| **dbPTM O-linked ∩ O-GlcNAcAtlas** | **5,824** | **34.9%** | **41.3%** |
| dbPTM N-linked ∩ N-GlyDE positives | 2,209 | 8.1% | 99.7% |
| dbPTM N-linked ∩ dbPTM O-linked | 1 | 0.0% | 0.0% |

The N/O separation is clean (1 shared site). The **O-GlcNAc containment** is not: O-GlcNAcylation
is a *subtype* of O-glycosylation, and dbPTM's broad O-linked bucket contains both.

### `annotation_discrepancies.parquet` — 63,017 candidates

| Class | Sites | Question for a validated latent |
|---|---|---|
| `single_source_despite_coverage` | 59,218 | missed by the other source, or spurious? |
| `containment_uncatalogued_oglcnac` | 3,577 | is this O-linked site actually O-GlcNAc? |
| `gold_negative_contradicted` | 222 | is this "verified unmodified" site actually modified? |

Only types with **two or more** contributing sources can show disagreement — hence
`max_disagreeing_sources = 1` throughout, and 0 for single-source types.

**Fourteen types show genuine disagreement**, led by ubiquitination (mean 0.395), sumoylation
(0.327), succinylation (0.268), butyrylation (0.227), N-glycosylation (0.226) and acetylation
(0.218), then propionylation (0.154), glutarylation (0.125), methylation_K (0.122), lactylation
(0.078), malonylation (0.076), neddylation (0.008), crotonylation (0.007) and methylation_R
(0.004). Every other type is exactly 0.000 — single-source, so disagreement is structurally
impossible. **Succinylation's 0.268 is the one that matters most for a headline claim**: CPLM and
dbPTM's own succinylation file disagree about roughly a quarter of its sites while both covering
the protein, which compounds the 90.65% crosstalk already noted in §3.

---

## 9. qPTM — the CF-26 replication set

Kept entirely separate from the main label files.

| Stage | Rows |
|---|---|
| raw | 11,482,553 |
| human only | 6,087,375 |
| reliability ≥ 2 | 5,398,801 |
| residue-unresolved dropped | −8,009 |
| chemistry validated | 3,372,856 |
| **after dedup** | **131,700** |

**Independence is limited and must be quoted.** At floor 2, **91.2%** of retained qPTM sites also
appear in dbPTM:

| Floor | Unique sites | % also in dbPTM |
|---|---|---|
| 0 | 422,510 | 62.9% |
| **2 (chosen)** | **186,525** | **91.2%** |
| 5 | 81,457 | 98.2% |

**qPTM covers only 6 types, and succinylation is not one of them** — one headline type has no
replication available.

qPTM headline-eligible types: N-glycosylation, acetylation, methylation_R, phosphorylation_ST,
phosphorylation_Y, sumoylation, ubiquitination.

---

## 10. Out-of-distribution — four species

| Species | Proteins | Sites (full) | Sites (strict, no human homolog) | Testable types, full (CF-22) | Testable types, strict |
|---|---|---|---|---|---|
| mouse | 15,280 | 191,577 | 37,405 (19.5%) | **16** | 12 |
| yeast | 6,229 | 89,583 | 77,183 (86.2%) | 6 | 6 |
| rat | 7,546 | 59,356 | 10,123 (17.1%) | 8 | 4 |
| **E. coli K-12** | 4,474 | 17,197 | 14,599 (84.9%) | 5 | 4 |

**35 of 200 (species, type) pairs are testable in the full view; 26 of 200 are still testable once
restricted to proteins with no human homolog** (`cd-hit-2d`, 40% identity, same threshold as the
main split). Full and strict counts are both in `T5_ood_applicability.csv`
(`testable`/`n_sites_species` vs. `testable_strict`/`n_sites_species_strict`).

**Mouse and rat lose most of their testable pairs under strict filtering** — both species are
close enough to human that most of their proteins have a whole-protein homolog there, so pairs
sitting just above the 200-site floor in the full view can fall well below it once restricted to
genuinely novel proteins. Two concrete cases: mouse `sumoylation` (212 full sites, `testable=True`)
falls to 36 strict sites (`testable_strict=False`); mouse `methylation_K` (218 → 20) shows the same
pattern. **Yeast and E. coli barely change** (86.2% / 84.9% retention) — most of their proteins were
never homologous to human to begin with, so the strict view is close to the full view for both.

**Use the strict count for a genuine species-novelty claim.** The full-view count answers "does the
latent generalise to this species at all" (still a legitimate, weaker claim); the strict count
answers "does it generalise to proteins with no whole-protein match in the human corpus" and is
the number that actually rules out "this is just re-detecting a human-like pattern."

Source coverage — measured, not assumed:

| Species | dbPTM | CPLM | qPTM | O-GlcNAcAtlas |
|---|---|---|---|---|
| mouse | ✓ | ✓ | ✓ | ✓ |
| rat | ✓ | ✓ | ✓ | ✓ |
| yeast | ✓ | ✓ | ✓ | ✓ |
| E. coli | ✓ | ✓ | ✗ | ✗ |

*(N-GlycositeAtlas and N-GlyDE are human-only and contribute nothing to OOD.)*

### E. coli is the sharpest cross-species test

| Type | E. coli (full) | E. coli (strict) | Human | Status |
|---|---|---|---|---|
| acetylation | 9,604 | 8,447 | 38,225 | testable, both views |
| **succinylation** | **4,047** | 3,354 | 3,627 | testable, both views — **more sites than human** |
| malonylation | 1,701 | 1,326 | 4,121 | testable, both views |
| phosphorylation_ST | 1,256 | 1,005 | 228,365 | testable, both views |
| phosphorylation_Y | 233 | 194 | 35,903 | testable full, **not strict** (194 < 200 floor) |
| **N-glycosylation** | **0** | 0 | 13,981 | **`absent_in_species`** |
| pupylation | 69 | 55 | 0 | bacterial-only, below floor either way |

E. coli's zero N-glycosylation is **biology, not latent failure** — bacteria do not perform it.
This is exactly why CF-22's matrix is required: without it the two are indistinguishable. E. coli
is also the species where full and strict views diverge least (84.9% retention), so its testable
set is the most stable of the four across both lenses — one more reason it's the sharpest test.

**Caveats.** The strict view controls for *whole-protein* homology, the same way CD-HIT does for
the human discovery/holdout split — it does not rule out a shorter shared motif around the
modified residue itself. Report strict-OOD results as *"no whole-protein homolog in the human
corpus,"* not as *"sequence-novel"* in an absolute sense. Yeast N-glycosylation is
high-mannose-dominated and diverges from mammalian; yeast O-GlcNAc is disputed, so its 330 sites
warrant caution regardless of which view is used.

---

## 11. Corpus and clustering

| | Measured |
|---|---|
| Proteins | 18,128 (gate: 15,000–20,000 ✓) |
| Length | mean 412, median 372, range 2–1,022 |
| Unique sequences | 18,050 (78 exact duplicates) |
| Clusters at 40% identity | 11,909 |
| Singletons | 9,110 clusters — **50.25% of proteins**, 76.50% of clusters |
| Mean / median / max cluster | 1.522 / 1 / 160 |

Healthy clustering — neither collapsed (one giant cluster) nor a no-op (all singletons). Largest
cluster is 0.9% of the corpus.

**Split quality:** achieved holdout 20.65% against a 20% target.

On **final (post-M3) counts**, the 20 headline types span **0.208–0.252** — 18 sit inside 0.20–0.25,
and the two that cross it do so marginally: methylation_K at 0.2520 and S-nitrosylation at 0.2522.
On **T3's raw, pre-filter counts** (which is what the greedy stratifier actually optimised, since
M2 must split before M3's filtering exists) the same 20 span **0.209–0.277**, with methylation_K
(0.2691), S-nitrosylation (0.2768) and glutathionylation (0.2552) above 0.25. Quote whichever is
appropriate, but state which — the two are not interchangeable.

**Pooled buckets: rows are not residues.** `K::other_PTM` contains **615 label rows over 516
distinct lysines** -- 63 residues carry two or more of the twelve sub-200-site lysine types
(butyrylation+lactylation 14x, butyrylation+glutarylation 11x, one lysine carrying five at once).
This is correct in the label file: they are genuinely distinct modifications and belong as
separate rows. But **any consumer counting a pooled bucket must count distinct (accession,
position) pairs, not rows** -- summing T2's per-type `positive_sites` gives 615, which
over-counts by 19.2%. Every other pooled bucket (C, E, N, Q, R, W, Y) is unaffected at 0.0%, and
all 21 above-floor types have exactly one row per (residue, type). See N2-CL#44 for the real bug
this caused downstream.

**19 types deviate >10 pp from target**, every one of them sub-200-site and none headline-eligible.
The tail is dominated by types with so few clusters that a single cluster decides the whole split:
serotonylation, S-cysteinylation and S-sulfhydration all land at 1.00, dietylphosphorylation 0.94,
biotinylation 0.89, propionylation 0.85, carboxymethylation 0.76, citrullination 0.72, with
glutarylation at 0.44 and lactylation at 0.38. Worst absolute deviation from target is 0.80.

---

## 12. Known limitations carried forward

1. **Crosstalk is extreme in the K stratum** — succinylation 90.7%, O-GlcNAcylation 69.0%. Two of
   three headline types are confounded.
2. **CPLM has no LP confidence filter** — the `lp` column is 100% null in the real download, so
   the Class II/III mask was unreachable and was removed outright (CL#50). CPLM sites are all
   retained as positives.
3. **dbPTM ubiquitination coordinates** — ~33% fail validation; dropped, not corrected.
4. **Isoform-recovered sites carry residual risk** — a shifted position can coincidentally land on
   a valid residue. Flagged `from_isoform_accession` for sensitivity runs.
5. **qPTM independence is limited** — 91.2% dbPTM overlap at floor 2.
6. **qPTM lacks succinylation.**
7. **Iodination contributes zero sites** — 19 sites, all in thyroglobulin (2,768 aa, over the cap).
8. **O-GlcNAc containment is mitigated, not resolved** — masking removes affected sites from the
   background rather than correctly classifying them.
9. **The stratified split targets raw, unfiltered counts** — M2 must split before M3's filtering
   exists. Tight for primary types (97–99% retention); T3 reports achieved fractions.
10. **E/W/Q pooled `other_PTM` buckets are themselves below the 200 floor.**
11. **OOD cross-species homology to human is now measured, not just flagged** — `cd-hit-2d`
    (40% identity, CL#51) distinguishes full OOD from strict (no human homolog) at both the
    site level (§10's retention table) and the (species, type) applicability level
    (`T5_ood_applicability.csv`'s `testable_strict`, CL#53). The residual limitation is narrower
    than before: the strict view controls for *whole-protein* homology only, not a shorter shared
    motif around the modified residue — see §10's caveat.
12. **No global identity threshold fully removes leakage** for residue-level PTM prediction —
    CD-HIT compares whole sequences, but a latent responds to ±7 residues of local context. The
    supportable claim is *"controls for whole-protein homology"*, not *"eliminates leakage"*.
13. **Labels are positives only** — negatives are derived at query time from
    `corpus.parquet` + `stratum_counts.json` + `exclusion_mask.parquet` +
    `gold_negatives_nglyco.parquet`. Matches the original pipeline, not the plan's stated schema.
    **N3 must be built against this convention.**
