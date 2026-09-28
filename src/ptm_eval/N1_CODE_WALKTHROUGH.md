# N1 Code Walkthrough — every step, in execution order

A line-by-line-level map of what the N1 notebook actually does, in the order it runs, with each
step tied back to the implementation plan. Read top to bottom and you will know exactly what
happens to a PTM site between "downloaded" and "written to parquet".

**Notation:** `[PLAN M2.4]` = implements plan step 4 of M2. `[NEW]` = not in the original.
`[FIX #n]` = changelog entry N1-CL#n.

---

# PART 0 — Foundations you need before reading the cells

## 0.1 UniProt vs Swiss-Prot, and why they anchor everything

**UniProt** is the umbrella protein database. It has two halves:

| Half | What it is | Size (human) | Used here? |
|---|---|---|---|
| **Swiss-Prot** | manually reviewed and curated by experts | ~20,000 | **yes — exclusively** |
| TrEMBL | automatically annotated, unreviewed | ~200,000 | no |

N1 uses **Swiss-Prot only** (`reviewed:true` in the query). This single choice drives almost
everything downstream:

**1. Swiss-Prot supplies the sequences.** Every PTM database reports a site as
`(accession, position)` — e.g. "P04075 position 39". That's just a coordinate; it carries no
amino acid. To know *what residue* is at position 39 of P04075, you need the sequence, and that
comes from Swiss-Prot. Without it, chemistry validation is impossible.

**2. Swiss-Prot defines the corpus, and therefore the denominator.** Your enrichment statistic
asks *"do latents fire on N-glycosylation sites more than on asparagines generally?"* The
"generally" is **every N in the corpus** — 263,374 of them. That number, *g<sub>ℓ</sub>*, comes
from counting residues in Swiss-Prot sequences. Change the corpus, change every enrichment value.

**3. Swiss-Prot is the membership test.** Any site whose accession isn't in the corpus is dropped
as `accession_not_in_corpus`. **This is the single largest filter in the pipeline** — bigger than
chemistry validation. CPLM loses 91,847 rows this way vs 7,591 to chemistry.

**4. Swiss-Prot's accession namespace is the join key.** All six PTM sources report UniProt
accessions, which is the only reason they can be merged at all.

So Swiss-Prot plays four roles at once: **sequence provider, background denominator, membership
filter, and join key.** The PTM databases supply only labels — the coordinates and the biology
come from Swiss-Prot.

> **Why `length ≤ 1022`:** ESM-2 650M's context limit, and both SAE families were trained at that
> cap. Not a choice. It's also why Iodination contributes zero sites — all 19 are in
> thyroglobulin, which is 2,768 aa.

## 0.2 How the data transforms across stages — the one-screen version

```
   UniProt/Swiss-Prot                 Six PTM databases
   (sequences)                        (coordinates + type labels)
        │                                       │
        │                                       │
   ┌────▼───────────────────────────────────────▼────┐
   │ M1  download both, verify, log SHA               │
   └────┬───────────────────────────────────────┬────┘
        │                                       │
   ┌────▼───────────────────────────────────────▼────┐
   │ M_E  measure only. Changes nothing.              │
   │      "what is actually in these files?"          │
   └────┬────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────┐
   │ M2  SEQUENCES → corpus            │   18,128 proteins
   │     count residues → g_ell        │   7,474,423 residues
   │     cluster → split               │   11,909 clusters, 80/20
   └────┬──────────────────────────────┘
        │ corpus.parquet  (accession → sequence, cluster, split)
        │
   ┌────▼──────────────────────────────┐
   │ M3  COORDINATES → labels          │   629,098 raw candidate rows
   │     validated against the corpus  │   → 498,619 confirmed sites
   └────┬──────────────────────────────┘
        │
        ├─► labels_stratified.parquet   (the main label set)
        ├─► labels_naive.parquet        (same sites, whole-corpus background)
        ├─► labels_qptm.parquet         (independent replication, CF-26)
        └─► labels_ood_<species>.parquet ×4
```

**The key idea in one sentence:** M2 turns *sequences* into a corpus and a background count;
M3 turns *coordinates* into validated labels **by checking them against that corpus**. M3 cannot
run before M2, because "is position 39 of P04075 an asparagine?" has no answer without the corpus.

| Stage | Input | Output | Unit of work |
|---|---|---|---|
| M1 | URLs | raw files | files |
| M_E | raw files | 8 diagnostic tables | measurements (no data changes) |
| M2 | Swiss-Prot TSV | corpus + g_ell + split | **proteins** |
| M3 | PTM DB files + corpus | validated label sets | **sites** |

---

# CELL 1 — Environment setup

1. `pip install` pyarrow, openpyxl, requests, tqdm, matplotlib.
2. `apt-get install cd-hit` — **not on Kaggle's base image**; M2 step 4 fails without it.
3. Sets two switches:
   - `MOCK_MODE` — `False` for real runs. `True` uses synthetic fixtures and needs no network.
   - `FORCE_REBUILD` — `False` reuses caches. Set `True` after editing parsing/filtering **logic**,
     because fingerprints hash *config*, not code.

---

# CELL 3 — M0: Configuration

Nothing executes here except definitions. Order matters because later definitions reference
earlier ones.

## 3.1 `Paths`
Resolves Kaggle vs local. Three zones with different lifetimes:

| Zone | Path | Lifetime |
|---|---|---|
| `ROOT` | `/kaggle/working` | persists as notebook output (this becomes the dataset) |
| `TEMP` | `/kaggle/temp` | scratch, discarded |
| `INPUT` | `/kaggle/input` | attached datasets, read-only `[FIX #1]` |

Creates `ROOT`, `TEMP`, `RAW`, `TABLES`, `FIGURES`, `EXPLORATION` at import time.

## 3.2 `Config` dataclass (frozen)
Only fields M1–M3 + M_E read. Key groups:

- **Corpus**: `max_protein_length=1022` `[PLAN M2.1]` — ESM-2 650M's hard ceiling, not a choice.
- **Label filters**: `min_sites_per_type=200`, `min_sites_headline=1000` `[PLAN M3.5]`.
- **`lp_class_positive`/`lp_class_masked`** — marked **INERT**: CPLM's `lp` column is 100% null in
  the real download, so this mask can never fire `[PLAN M3.3b]`.
- **`qptm_reliability_floor=2`** `[NEW #22/#24]` — qPTM's only quality signal.
- **`ood_species`** `[NEW #35]` — four species with per-source labels. A `None` label means that
  source genuinely doesn't cover that species (qPTM has no E. coli).
- **`ood_min_sites_per_type=200`** `[NEW #36]` — CF-22 applicability threshold, applied to both
  the full and strict (homology-corrected) views `[NEW #53]`.
- **Redundancy**: `cdhit_identity=0.40`, `cdhit_cluster_before_split=True` `[PLAN M2.4]`.
- **Split**: `holdout_fraction=0.20`, `holdout_seed=42`. `holdout_stratify_by` is **descriptive
  only** — written to the manifest, read by nothing.
- **`stratum_residues`** — ten strata: K, ST, N, C, R, Y + **E, M, W, Q** `[NEW #14]`.
- **`ptm_type_to_stratum`** — ~47 canonical types → stratum. **This dict is the gate**: a type not
  in it cannot be analysed, because it has no background denominator.
- **`primary_types`** — the three headline types.

## 3.3 Type canonicalisation `[FIX #17: moved here from M3]`
Moved so M_E (which runs *before* M3) uses the same mapping rather than a second copy.

- **`PTM_TYPE_ALIASES`** — normalised source spelling → canonical key. Includes
  `"ubiquitylation" → "ubiquitination"` `[FIX #20]`, qPTM's spelling.
- **`PTM_TYPE_RESIDUE_RESOLVED`** `[NEW #20]` — labels covering **more than one chemistry**,
  resolvable only by the actual residue:

  | label | N | S/T | Y | K | R |
  |---|---|---|---|---|---|
  | `glycosylation` | N-glyco | O-glyco | — | — | — |
  | `phosphorylation` | — | phospho_ST | phospho_Y | — | — |
  | `methylation` | — | — | — | methyl_K | methyl_R |

- **`normalize_ptm_type(raw, known_types, residue)`** — resolution order:
  1. residue-resolved table (only if `residue` supplied **and** the label is multi-chemistry)
  2. alias table
  3. case/format variant of an existing canonical key
  4. unchanged → caller decides
  With `residue=None` behaviour is byte-identical to the old alias-only lookup, so this is
  strictly additive.
- **`type_to_stratum(raw)`** — convenience for M_E reporting; returns `"UNMAPPED"` if unknown.

### Worked example — `normalize_ptm_type` step by step

Take a real qPTM row: **`type="Glycosylation"`, accession `P04075`, position 39**, and suppose
the corpus sequence has **`N`** at position 39.

| Step | What happens | Config variable used |
|---|---|---|
| 1 | `_normalize_type_key("Glycosylation")` → lowercase, `_`→`-`, spaces→`-` ⇒ **`"glycosylation"`** | — |
| 2 | Is `"glycosylation"` in `PTM_TYPE_RESIDUE_RESOLVED`? **Yes.** Look up residue `"N"` in its sub-dict `{N: N-glycosylation, S: O-glycosylation, T: O-glycosylation}` ⇒ **`"N-glycosylation"`** | `PTM_TYPE_RESIDUE_RESOLVED` |
| 3 | Return immediately — steps 4–5 are skipped | — |
| 4 | *(skipped)* alias lookup | `PTM_TYPE_ALIASES` |
| 5 | *(skipped)* case/format variant match | `ptm_type_to_stratum` keys |

Then the **caller** (`canonicalize_types`) checks membership:
`"N-glycosylation" in CFG.ptm_type_to_stratum` → **yes**, stratum `"N"`. The row survives.

**Same row, different residue.** If position 39 held `S` instead, step 2 returns
`"O-glycosylation"` → stratum `"ST"`. **One source label, two destinations, decided purely by the
residue.** That is the whole point of `[NEW #20]`.

**Three ways this can fail, and what each does:**

| Case | Step 2 | Step 4 (alias) | Step 5 | Outcome |
|---|---|---|---|---|
| residue is `L` | sub-dict miss → fall through | `"glycosylation"` not in aliases | no canonical key matches | returned unchanged → **dropped** as residue-unresolved `[FIX #39]` |
| accession not in corpus (`residue=None`) | skipped entirely | same | same | same — **dropped** |
| `type="Fictionylation"` | not multi-chemistry | not in aliases | no match | returned unchanged → **RAISES** |

The last two rows look identical but are treated oppositely, and that distinction *is* fix #39:
a known label whose residue can't be determined is benign (step 2 would drop it anyway); an
unknown label means real data is silently vanishing and must stop the run.

**Contrast — a plain alias case.** `type="Ubiquitylation"`:
step 2 → not in `PTM_TYPE_RESIDUE_RESOLVED` (ubiquitination is lysine-only, no ambiguity);
step 4 → `PTM_TYPE_ALIASES["ubiquitylation"] = "ubiquitination"` ✓. Residue never consulted.

### Step 1 vs Step 5 — they normalise *different things*

This is the most confusing part of the function, so precisely:

| | What gets normalised | Against what |
|---|---|---|
| **Step 1** | the **input string** from the source file | nothing — it just rewrites the input |
| **Step 5** | **every canonical key** in `ptm_type_to_stratum` | compares each to the already-normalised input |

**Why step 1 doesn't make step 4 succeed.** Step 1 does turn `"Carboxylation"` into
`"carboxylation"` — correct. But step 4 then asks *"is `"carboxylation"` a **key in
`PTM_TYPE_ALIASES`**?"* and the answer is **no** (verified: `'carboxylation' in
PTM_TYPE_ALIASES → False`).

`PTM_TYPE_ALIASES` is a hand-written **rename table**. It only contains entries where the source's
name **differs** from the canonical name — `"n-linked-glycosylation" → "N-glycosylation"`,
`"sulfhydration" → "S-sulfhydration"`, `"ubiquitylation" → "ubiquitination"`. Nobody wrote an
alias for `carboxylation` because **no rename is needed**: the canonical name already *is*
`"carboxylation"`, CPLM just capitalises it.

Step 5 exists to handle exactly that class without forcing ~47 pointless identity entries into
the alias table. It normalises each canonical key and checks for equality — so it can match
`"Carboxylation"` to `"carboxylation"`, but it **cannot invent** a mapping, because it only ever
returns a key that already exists in `ptm_type_to_stratum`.

> **Cosmetic inconsistency worth knowing:** the alias table *does* contain some redundant identity
> entries (`"succinylation": "succinylation"`, `"acetylation": "acetylation"`,
> `"O-GlcNAcylation": "O-GlcNAcylation"`). Those resolve at step 4 instead of step 5. Both give
> the same answer — it's just that the table was built incrementally. Harmless.

### Every possible case, traced

Verified by running each input through the real function:

| # | Case | Example input | Residue | Fires at | Result |
|---|---|---|---|---|---|
| **A** | multi-chemistry, residue known **and listed** | `Glycosylation` | `N` | **2** | `N-glycosylation` |
| | | `Glycosylation` | `S` | **2** | `O-glycosylation` |
| | | `Phosphorylation` | `Y` | **2** | `phosphorylation_Y` |
| | | `Methylation` | `R` | **2** | `methylation_R` |
| **B** | multi-chemistry, residue known but **not listed** | `Glycosylation` | `L` | none | unchanged → **DROP** |
| **C** | multi-chemistry, residue **undeterminable** | `Glycosylation` | `None` | none | unchanged → **DROP** |
| **D** | rename — source name ≠ canonical | `N-linked Glycosylation` | — | **4** | `N-glycosylation` |
| **E** | synonym | `Ubiquitylation` | — | **4** | `ubiquitination` |
| **F** | missing prefix | `Sulfhydration` | — | **4** | `S-sulfhydration` |
| **G** | redundant identity alias | `Succinylation` | — | **4** | `succinylation` |
| **H** | **case only**, no alias needed | `Carboxylation` | — | **5** | `carboxylation` |
| **I** | spacing/hyphenation | `Gamma-carboxyglutamic acid` | — | **5** | `gamma-carboxyglutamic-acid` |
| **J** | genuinely unknown | `Fictionylation` | — | none | unchanged → **RAISE** |

**Cases B, C and J all return the input unchanged — but are treated oppositely.** The caller
(`canonicalize_types`) distinguishes them by asking *"was the normalised key in
`PTM_TYPE_RESIDUE_RESOLVED`?"*:

- **Yes** (B, C) → this is a *known* label we simply couldn't resolve for this row → **drop and
  count** as `n_residue_unresolved_dropped`. Step 2 would have dropped it anyway.
- **No** (J) → a label nothing knows → **raise**, because real data is silently vanishing.

That single distinction *is* fix `[#39]` — the bug that crashed your first full run on 8,009 qPTM
rows in cases B and C.

### `normalize_ptm_type` vs `canonicalize_types` — the difference

These are constantly confused, so explicitly:

| | `normalize_ptm_type` | `canonicalize_types` |
|---|---|---|
| **Lives in** | M0 (config cell) | M3 |
| **Operates on** | **one string** | **a whole DataFrame** |
| **Signature** | `(raw_type, known_types, residue) -> str` | `(df, cfg, corpus_seq) -> (df, log)` |
| **Knows about residues?** | only if you hand it one | **looks them up itself** from `corpus_seq` |
| **Can raise?** | never | **yes** — on genuinely unknown labels |
| **Returns** | a canonical type name | a filtered df + a step-log entry |

`canonicalize_types` is the **cascade step**; `normalize_ptm_type` is the **lookup function it
calls per row**. In order, `canonicalize_types`:

1. For every row, looks up the residue: `corpus_seq[accession][position-1]`, or `None` if the
   accession is absent or the position out of range.
2. Calls `normalize_ptm_type(type, known_types, residue)` per row — **memoised** on
   `(raw_type, residue)` `[FIX #38]`, since qPTM alone is ~5.4M rows.
3. Splits whatever is still unmapped into two groups: residue-unresolvable (**drop + count**) and
   genuinely unknown (**leave in, so the caller raises**) `[FIX #39]`.
4. Returns the df plus `{n_rows, n_unmapped_rows, n_residue_unresolved_dropped, unmapped_types,
   residue_aware}`.

## 3.4 Diagnostic printing `[NEW #42]`
`show_table()` (uses `to_string()` — no truncation), `show_json()`, `cluster_size_report()`.
Pandas display options widened. **Why:** the saved `.ipynb` is the only artefact reviewable
without the output dataset; `display()` truncates silently.

## 3.5 Caching layer
- `fingerprint(*parts)` — SHA-256 of JSON-serialised objects, first 16 hex chars.
- `find_in_inputs` / `find_cached` — search order: `ROOT` (recursive) → attached input datasets →
  caller-supplied dirs.
- `ensure_artifact(filename, build_fn, expected_fingerprint, force)` — cache-or-build. A hit in an
  input dataset is **copied into `ROOT`** so every downstream path assumption holds.
- **`config_fingerprint(*extra, cfg)`** — hashes every output-affecting config field. `[FIX #47]`
  now includes `stratum_residues`, `qptm_reliability_floor` and `ood_min_sites_per_type`, which
  were missing — editing any of them previously reused stale cache silently. Sets are sorted to
  lists first so the hash is stable across runs.

## 3.6 `Manifest`
`sha256_file()` + `Manifest.log(source, url, dest, notes)` records SHA, byte size and access
timestamp per artefact `[PLAN M1 emits]`.

---

# CELL 5 — M1: Acquisition (definitions)

## 5.1 Source constants
- `DBPTM_TYPE_URLS` — **37 types** (was 13) `[NEW #13/#14]`.
- `CPLM_BASE` + `CPLM_SPECIES_HUMAN = "Homo sapiens"` — the human corpus uses this **only**.
- `UNIPROT_STREAM` + `UNIPROT_FEATURE_FIELDS`.
- `QPTM_DIRECT_URL` `[NEW #6]` — real bulk link with a personal access code. **Don't share the
  notebook publicly with this intact.**
- `OGLCNAC_URLS` (Dataset-I and II), `GLYCOSITE_ATLAS_URL`, `STACKGLYEMBED_BASE`.

## 5.2 Download helpers
- `download_file` — streaming, raises on non-2xx.
- `_download_and_extract_zip_member` `[NEW #5]` — downloads a ZIP, streams its **largest member**
  straight to disk. Needed for qPTM (1.55 GB) and only qPTM.
- `_resolve_raw(rel_path)` — checks `RAW` first, then any attached input dataset by filename.

## 5.3 `acquire_*` functions
Each follows the same shape: resolve cache → download or mock → `manifest.log()` → return path.

| Function | Notes |
|---|---|
| `acquire_uniprot_reviewed` | organism-parameterised — this is why OOD proteomes are free |
| `acquire_uniprot_swissprot` | thin wrapper, human (9606) |
| `acquire_cplm` | Section 1 only; **never** Section 2 (~45 GB) |
| `acquire_qptm` | uses the ZIP helper; raises if `QPTM_DIRECT_URL` unset |
| `acquire_dbptm` | loops all 37 types |
| `acquire_oglcnac_atlas` | Dataset-I **and** II — II drives the mask |
| `acquire_glycosite_atlas` | XLSX, human-only |
| `acquire_glyde` | train+test, `_original` variant; **only Gold-tier negative source** |
| `acquire_ood_proteomes` `[NEW #25]` | four reviewed proteomes |
| `acquire_cplm_ood` `[NEW #35]` | four **strain-exact** names; failure is non-fatal |

> **Not here:** ESM-2/SAE acquisition, removed to N2 `[#4]`.

## 5.4 Parsers — raw file → `(accession, position, type, ...)`
- `_match_column(columns, candidates)` — substring matcher. `[FIX #8]` added `"ptm"` for qPTM's
  real header.
- `_decode_bytes_robust` — utf-8 → cp1252 → latin-1. latin-1 never raises, so it's a true fallback.
- `_read_csv_robust`, `_read_flatfile_tsv` — the latter **sniffs content**: tar → gzip → plain.
  dbPTM's `.gz` links are actually gzip-wrapped TAR archives.
- `parse_cplm_zip` — **no header row**; read positionally against `CPLM_POSITIONAL_COLUMNS`.
- `parse_dbptm_flatfile` — no header; `evidence` hard-set to `"experimental"` (this *is* the
  experimental endpoint). `[NEW #25]` keeps `species_suffix` from `entry_name` — dbPTM's only
  species signal.
- `parse_qptm_flatfile` — `[FIX #8]` column matcher; `[NEW #22]` carries `reliability`.
- `parse_oglcnac_atlas_flatfile(path, species="human")` `[NEW #35]` — filters to species **and**
  `accession_source == "uniprot"`. Uses `position_in_protein`, **not** `position_in_peptide`
  (silent large off-by-N if swapped).
- `parse_glycosite_atlas_flatfile` — XLSX with a **title row above the header**; header located
  dynamically by `_find_excel_header_row`.
- `parse_stackglyembed_fasta_pairs` / `parse_glyde_flatfile` — two lines per protein, odd line
  `accession,label,pos[,label,pos...]`.

## 5.5 Mock fixtures
One `_write_mock_*` per source. Two are deliberately hostile: `_write_mock_qptm` emits the real
16-column header and all four species; `_write_mock_oglcnac` is written **cp1252 with a non-UTF-8
byte** `[#19]` — reproducing the encoding crash a pure-ASCII fixture hid.

## 5.6 `run_m1(paths, mock, force, ood_species)` — **first real execution**
1. `tqdm` over 7 human sources → `resolved` dict (source → path or dict-of-paths).
2. qPTM wrapped in try/except: a missing URL prints `HELD` rather than aborting.
3. If `ood_species`: acquire 4 proteomes + 4 CPLM species.
4. Builds `cplm_human_compact.parquet` via `ensure_artifact`, fingerprinted on the CPLM zip's SHA.
5. `manifest.save()` → `manifest.json`.

**Returns `(manifest, resolved)`. `resolved` is the source of truth for paths** — downstream never
hard-codes `RAW / "file"`, because a cache hit may live inside an input dataset.

---

# CELL 6 — M1 execution
```
manifest, resolved = run_m1(Paths, mock=MOCK_MODE, force=FORCE_REBUILD, ood_species=CFG.ood_species)
report_raw_download_sizes(resolved)
```
Size report `[#12]` covers 9 source groups `[FIX #41]` — OOD downloads were previously omitted
from the total.

**Measured:** 54 artefacts, 1,679 MB (all sources, human + OOD).

---

# CELL 8 — M_E: Exploration (definitions) `[ENTIRE STAGE NEW #18/#23]`

Runs **between M1 and M2**, before anything is filtered. Exists because several plan thresholds
were set from assumption before anyone looked at the data. **Nine of twelve substantive M2/M3
revisions came from this stage.**

- `profile_dataframe(df, label)` — shape, per-column dtype/nulls/cardinality, memory, 3 sample rows.
- `value_counts_report` / `numeric_summary` — categorical and numeric breakdowns.
### `load_human_swissprot_lookup` — what it does and why it exists

```python
{"P04075": "MPYQYPALT...", "P07814": "MATLYKALS...", ...}
```

A plain `{accession: sequence}` dict built by reading M1's **raw Swiss-Prot TSV**.

**Why not just use the corpus?** M_E runs *before* M2 — `corpus.parquet` doesn't exist yet. This
is deliberate: M_E's job is to tell you what's in the data *before* you commit to corpus-building
decisions.

**Consequence to keep in mind:** the raw download includes proteins M2 will later drop
(>1,022 aa). So M_E's "human" counts are very slightly **more generous** than M3's final counts.
That is expected, not a discrepancy — it's why M_E's `n_human_sites` for
N-GlycositeAtlas is 25,181 while M3's post-corpus count is 24,829.

It is used for exactly two things: the **membership test** (`accession in lookup`) and the
**residue lookup** (`lookup[acc][pos-1]`).

### `_classify_sites` → `_suggest_stratum` → `chemistry_failure_breakdown` — one chain

These three are a pipeline, not independent utilities. `chemistry_failure_breakdown` calls both
of the others; you never call the inner two directly.

**Step 1 — `_classify_sites(accessions, positions, lookup)`** walks every site once and sorts it
into one of three buckets:

| Bucket | Condition | Meaning |
|---|---|---|
| `n_non_human` | accession not in lookup | wrong species, or not reviewed |
| `n_out_of_range` | position < 1 or > len(seq) | **coordinate/isoform problem** |
| `residue_counts` | in range | a tally like `{K: 4151, R: 87, ...}` |

Returns `(n_non_human, n_out_of_range, residue_counts)`.

**Step 2 — `_suggest_stratum(residue_counts)`** answers *"which stratum does this type's data
actually look like?"* — ignoring what the config claims. It loops over **every** stratum in
`CFG.stratum_residues` and computes what fraction of sites that stratum's residues would capture:

Worked example, dbPTM `Methylation`, real counts `{K: 1874, R: 4151, G: 55, C: 29, Q: 27, ...}`,
total 6,269:

| Stratum | Its residues | Captured | Fraction |
|---|---|---|---|
| K | {K} | 1,874 | 29.9% |
| **R** | **{R}** | **4,151** | **66.3%** ← highest, wins |
| C | {C} | 29 | 0.5% |
| Q | {Q} | 27 | 0.4% |
| ST | {S,T} | ~20 | 0.3% |

Returns `("R", 66.3)`.

**Step 3 — `chemistry_failure_breakdown`** assembles the verdict per (dataset, type):

```python
stratum        = type_to_stratum(ptm_type)          # what the CONFIG says
valid          = CFG.stratum_residues[stratum]      # residues that implies
n_non_human, n_oor, residue_counts = _classify_sites(...)
n_ok           = sum(residue_counts[r] for r in valid)
suggested, pct = _suggest_stratum(residue_counts)   # what the DATA says
mapping_looks_wrong = (suggested != stratum and pct > 60)
```

For dbPTM Methylation: config says `K`, data says `R` at 66.3% → **`mapping_looks_wrong = True`**.
That single flag is what surfaced the residue-collapse bug. Two types tripped it on your real
run: dbPTM `Methylation` and qPTM `Glycosylation`.

**Why `>60%` and not "just different":** a type genuinely spread across several residues (dbPTM
`ADP-ribosylation`: S 30, E 18, C 12, D 8, K 5) has no single stratum that explains it. Its best
suggestion is `ST` at 35.7% — below threshold, correctly **not** flagged, because reassigning it
wouldn't help.
*(the `_classify_sites` → `_suggest_stratum` → `chemistry_failure_breakdown` chain is
explained in full above)*
- `type_coverage_report` — per (source, type): sites, accessions, **% human**, chemistry-match rate.
- `isoform_suffix_audit` — how many unmatched accessions are recoverable by stripping `-2`.
- `glyde_negative_contradiction_check` — do gold negatives appear as positives elsewhere?
- `dbptm_species_suffix_audit` — real species mixture per dbPTM type.
- `qptm_reliability_sensitivity` — retention **and dbPTM overlap** at every floor 0–5.
- `cross_dataset_overlap` — pairwise site overlap between look-alike sources.
- `run_me(resolved)` — runs all of the above, emits 8 `TE_*` tables.

---

# CELL 9 — M_E execution
`TE = run_me(resolved)`. Everything printed **and** saved. Peak memory here (~11 GB) — qPTM raw
8.3 GB + parsed 2.7 GB.

---

# CELL 11 — M2: Corpus (definitions)

- **`filter_swissprot`** `[PLAN M2.1]` — reviewed human, length ≤ 1,022. Re-asserts the filter even
  though the UniProt query already applies it.
- **`compute_stratum_counts(corpus, stratum_residues)`** `[PLAN M2.2/2.3]` — **exact** counts, not
  sampled; every enrichment value depends on $g_\\ell$ being exact. `[FIX #15]` takes strata as a
  parameter — it used to duplicate the config dict, which would silently go stale.
### `run_cdhit(corpus, identity=0.40)` `[PLAN M2.4]`

**What CD-HIT is for.** Homologous proteins share sequence *and* PTM sites. If one ends up in
discovery and its 40%-identical twin in holdout, your "held-out" score is inflated — the model
effectively saw it already. CD-HIT groups similar proteins into **clusters**, and the split is
then made **cluster-by-cluster**, so relatives always stay on the same side.

**How CD-HIT works, and what "word size" means.** Comparing 18,128 sequences pairwise would be
18,128² ≈ 330 million alignments. CD-HIT avoids that with a shortcut: chop each sequence into
overlapping **k-mers** ("words") of length `n`. `MKSTY` at n=2 gives `MK, KS, ST, TY`. Two
sequences that share almost no words cannot possibly be 40% identical, so they're never aligned.

Lower identity thresholds need **shorter** words. At 40% identity, matching stretches are short,
so long words would miss real homologs. CD-HIT's documented bands:

| Identity | Word size `n` |
|---|---|
| 0.70–1.00 | 5 |
| 0.60–0.70 | 4 |
| 0.50–0.60 | 3 |
| **0.40–0.50** | **2** ← ours |

So at 40% we use **n = 2** (dipeptides).

**Now the two silent behaviours the code works around.** Both cause CD-HIT to *drop proteins
without telling you* — and a dropped protein has no cluster, no split, and would crash step 6.

> **Behaviour 1 — the `-l` flag.** CD-HIT has a "throw-away length" parameter, default **10**:
> any sequence of 10 residues or fewer is discarded with no warning. Your corpus has proteins as
> short as **2 aa**, so this would silently delete them.
> **Fix:** pass `-l` explicitly, set to `n-1` = **1**. (Determined empirically against cd-hit
> 4.8.1: `n-2` is rejected with *"Fatal Error: Too short -l, redefine it"*, so `n-1` is the lowest
> value CD-HIT will accept for this word size.)

> **Behaviour 2 — sequences shorter than the word size.** A 1-residue protein cannot contain a
> single 2-mer. No `-l` setting fixes this: with zero words, CD-HIT has nothing to index it by, so
> it can never be clustered at this word length by any configuration.
> **Fix:** don't send them to CD-HIT at all. Sequences with `len < n` are filtered out beforehand
> and each is assigned **its own singleton cluster** afterwards — which is the correct answer
> anyway. A protein too short to compare to anything is, by definition, redundant with nothing.

**Then:** run CD-HIT, parse its `.clstr` output file (lines beginning `>Cluster N`, followed by
member lines), and build `{accession: cluster_id}`. Short sequences are appended with fresh
cluster ids continuing from the maximum CD-HIT produced.

**Measured on your run:** 18,128 proteins → 11,909 clusters. 50.25% singletons, mean size 1.522,
largest cluster 160. Healthy — a collapsed clustering (one giant cluster) or a no-op (all
singletons) would both produce a plausible-looking *count*, which is exactly why
`cluster_size_report` `[NEW #42]` prints the distribution.

### Which threshold should you actually use?

**Recommendation: keep 40% as primary; run 30% as a sensitivity check only if a headline result
looks homology-driven.** Reasoning, with the measured evidence:

**The three published precedents, and where 40% sits:**

| Threshold | Precedent | Task |
|---|---|---|
| 50% | ProtSyntax | residue-level PTM windows — **closest analogue to yours** |
| **40%** | COMPASS-PTM | protein-level split, also a PTM task |
| 30% | InterProt | Swiss-Prot family evaluation |

40% is **stricter than the closest task analogue** and matches the other PTM precedent. That is a
defensible primary setting, and it's what you instructed.

**What 40% costs you: nothing measurable.** The risk of a tighter threshold is collapsing the
corpus into too few clusters, starving the holdout. Your run shows that didn't happen:

- 11,909 clusters from 18,128 proteins — plenty of independent units
- 50.25% singletons — half the corpus has no homolog at 40% at all
- largest cluster 160 (0.9% of the corpus) — no runaway cluster
- **all 20 headline types are `v12b_eligible=True`** — ≥200 sites in *both* partitions

So tightening to 40% didn't cost you a single headline type's holdout validation.

**Why not go to 30%?** It would merge more distant homologs, shrinking the effective number of
independent clusters, for a leakage risk that 40% already addresses. There's no evidence in your
run that 40% is leaking. Going tighter without evidence trades real statistical power for a
hypothetical gain.

**Why the sensitivity re-run is cheap.** M6 stores per-residue histograms. Re-clustering at 30%,
re-splitting, and re-tallying those histograms needs **no new GPU inference** — only M2 and the
downstream tallies re-run. That is why the plan treats it as a cheap check rather than a second
full pipeline.

> ### The honest limitation — worth stating in the thesis
>
> **No global identity threshold fully removes leakage for residue-level PTM prediction.** CD-HIT
> compares whole sequences, but a PTM latent responds to **local context** — roughly ±7 residues
> around the site. Two proteins at 25% global identity can still share a near-identical 15-residue
> motif around a modified lysine. Clustering at 40%, 30%, or even 20% would not separate them.
>
> So the correct claim is *"the split controls for whole-protein homology"*, **not** *"the split
> eliminates leakage"*. A motif-level split (clustering the flanking windows themselves rather
> than the proteins) would address this directly — but it would break the corpus design, since the
> corpus must cover **every** protein to supply *g<sub>ℓ</sub>*, including the ~10,000 with no
> annotated site. Worth naming as a limitation rather than silently claiming more than the method
> delivers.
- **`raw_type_hits_by_accession`** `[FIX #26/#40]` — lightweight unfiltered per-accession type
  counts to drive stratification. Now keeps **every canonical type**, residue-aware.
- **`iterative_stratified_split`** `[PLAN M2.5]` — greedy multi-label stratification (Sechidis et
  al. 2011). Clusters with primary-type sites are processed first, assigned to whichever partition
  reduces summed squared shortfall from target; unlabelled clusters split by plain ratio.
  **No optimality guarantee** — which is why achieved fractions must be reported.
- **`build_corpus`** — orchestrates 1→5. Parses raw sources **before** CD-HIT so a parsing bug
  fails in seconds, not after clustering. Emits `T3_split_composition` `[NEW #26]`.
- **`save_corpus_artefacts`** — `corpus.parquet`, `stratum_counts.json`, `split_manifest.json`.
- **`run_m2`** — cache-aware wrapper; the three artefacts are cached and invalidated **as one
  unit** because they must stay mutually consistent.
- **`build_ood_corpus` / `run_m2_ood`** `[NEW #25]` — same length filter and same exact stratum
  counting; **no CD-HIT, no split** (an OOD species is held out by construction).

---

# CELL 12 — M2 execution

1. Registers **every positive-bearing source** `[FIX #40]`: all 37 dbPTM types (lambdas bind
   `t=_t` as a default arg — a bare closure would label every file with the last type),
   O-GlcNAcAtlas Dataset-I, N-GlycositeAtlas, CPLM. qPTM is registered for **cache fingerprinting
   only**, with no parser.
2. `run_m2(...)` → corpus, gate_report, cache info.
3. Prints stratum counts, **CD-HIT cluster diagnostics** `[NEW #42]`, gate report, full T3.
4. Flags any type deviating >10 pp from target.
5. `run_m2_ood(...)` → 4 OOD corpora + per-species $g_\\ell$ table.

**Measured:** 18,128 proteins · 11,909 clusters · holdout 0.2065 · **44 types stratified** · 50.25%
singletons · max cluster 160.

---

# CELL 14 — M3: Labels (definitions)

Execution order inside `run_m3` is **not** the definition order. Follow this:

### Step 0a — `strip_isoform_suffix` `[NEW #27]`
`P12345-2` → `P12345` when the stripped form exists in the corpus. Flags `from_isoform_accession`.
**Risk:** position refers to the *isoform's* numbering. Chemistry validation is the net, but a
shifted position can coincidentally land on a valid residue — hence the flag.

### Step 0b — `canonicalize_types(df, cfg, corpus_seq)` `[NEW #20]`
Looks up the **actual residue** first, then assigns the type. `[FIX #38]` memoised on
`(raw_type, residue)` — qPTM alone is ~5.4M rows × ~50 key scans uncached.
`[FIX #39]` distinguishes two failure modes:

| Situation | Action |
|---|---|
| label IS residue-resolvable, residue undeterminable | **drop**, count as `n_residue_unresolved_dropped` |
| label unknown to every table | **raise** |

### Step 1 — `step1_evidence_filter` `[PLAN M3.1]`
Keeps MS/MS + low-throughput; drops "by similarity"/"potential"/"probable". Unrecognised evidence
defaults to **drop** (conservative). Sources with no evidence column pass through unchanged.
**Measured: drops zero rows from every source** — every source used is already experimentally
filtered at origin.

### Step 2 — `step2_validate_sites` `[PLAN M3.2]`
Residue at (accession, 1-based position) must match the type's chemistry. Failures counted in
three buckets: `accession_not_in_corpus`, `position_out_of_range`, `residue_mismatch`.
**Measured:** corpus membership dominates — CPLM 91,847 of 99,727 drops.

### Step 3 — Ambiguity masking `[PLAN M3.3]` — four implementations
- **3a `step3a_oglcnac_mask`** — Dataset-II → mask. Reason
  `oglcnac_atlas_dataset_ii_ambiguous`. **2,805 sites.**
- **3b `step3b_cplm_lp_mask`** — LP Class II/III/IV → mask. **INERT** (lp 100% null); warns.
  `[FIX #44]` returns a real bool, not the string `"False"`.
- **3c `step3c_oglycosylation_ambiguity_mask`** `[NEW #21]` — dbPTM O-linked sites **not**
  confirmed in the Atlas → masked **for O-GlcNAcylation only**. Reason
  `dbptm_o_linked_glyco_may_be_uncatalogued_o_glcnac`. **3,577 sites.**
- **3d** — the same mask is passed into the qPTM cascade `[NEW #32]`.

> **CRITICAL, unchanged from plan:** masking is a **statistics-layer** operation. The full intact
> sequence still goes to ESM-2; masked positions are simply not tallied into any contingency cell.
> Deleting residues would shift every downstream position.

### Step 4 — `step4_dedup` `[PLAN M3.4]`
Merge on (accession, position, type); `source_count` = number of independent sources.
**Measured: 629,098 → 498,619.** Max `source_count` is **2**, structurally: dbPTM ships one file
per type, so a K-site can only be corroborated by CPLM + one dbPTM file.

### Step 4b — `build_site_agreement` `[NEW #28]`
Per site: which sources report it, which sources **cover (protein, type)**, and the disagreement
count. `[FIX #43]` covering is **type-aware** — the type-blind version reported 97% of sites as
disagreeing because a phospho file "covered" any protein with a phospho site.
`[FIX #37]` source labels tracked in lockstep (`zip` truncation made the old length guard
unfalsifiable).

### Step 5 — `step5_min_frequency` `[PLAN M3.5]`
<200 → pooled to `<stratum>::other_PTM`; ≥1,000 → `headline_eligible`.
`[FIX #45]` in naive mode `df["stratum"]` is now set to `"ALL"` — previously the placeholder was
set on the counts table only, making `labels_naive` a byte-identical copy of `labels_stratified`.

### Step 6 — `step6_inherit_split` `[PLAN M3.6]`
**A join, not a second clustering run.** Attaches `cluster_id`/`split` from the corpus. Raises if
any site has no assignment.

### Post-cascade builders
- `crosstalk_flags` — residues carrying >1 type.
- `_enrich_label_schema` `[NEW #33]` — adds `residue` and `crosstalk`.
- `build_gold_negatives_nglyde` `[FIX #29]` — returns `(clean, contradicted, report)`.
  **Measured: 222 of 1,310 (16.95%) removed, 1,088 clean.**
- `build_annotation_discrepancies` `[NEW #46]` — unifies three disagreement classes into the N4
  adjudication set.
- `build_T4_type_survival` `[NEW #31]` — per-type waterfall. `[FIX #44]` excludes mask-only
  sources from the early stages.
- `build_qptm_labels` `[PLAN CF-26]` — parallel cascade: species filter (**`"Human"`, not
  `"Homo sapiens"`** `[FIX #11]`) → reliability floor → canonicalise → validate → **mask** → dedup
  → pool → inherit split.
- `build_ood_labels` `[NEW #25/#35]` — same cascade minus split inheritance.
- `build_species_type_applicability` `[NEW #36]` — CF-22 matrix. `[FIX #53]` adds
  `n_sites_species_strict`/`testable_strict`, the same 200-site test restricted to
  `homologous_to_human == False` sites (CL#51) — `None`, not `False`, if that species never got a
  `homologous_to_human` column at all.
- `run_m3` — orchestrates everything above.
- `run_m3_cached` — fingerprint on config + corpus + every source's content hash.
- `build_T1` / `build_T2` / `build_F2` — plan tables. `[FIX #48]` F2 now has all five panels
  (a)–(e); panel (d), three-tier negative counts, was never implemented.

---

# CELL 15 — M3 execution

### Why `raw_sources` is built again here, when M2 already parsed these files

This looks redundant and isn't. **M2 and M3 need different columns for different jobs.**

| | M2 (`parsers`) | M3 (`raw_sources`) |
|---|---|---|
| Columns | `accession, position, type` **only** | `+ evidence, lp, pmid, species_suffix` |
| Purpose | count sites per protein per type, to **balance the split** | run the **eight-step cascade** |
| Needs `evidence`? | no — no filtering happens | **yes** — step 1 |
| Needs `lp`? | no | **yes** — step 3b |
| Output | `{accession: {type: count}}` — counts, discarded after | validated label rows — the product |

M2 is doing arithmetic: *"protein P1 has 3 succinylation sites"* — enough to balance clusters
across discovery/holdout. It deliberately takes a **3-column slice** because the split doesn't
care about evidence tiers or localisation probabilities, and slicing keeps the memory footprint
down on a 2.2M-row dbPTM parse.

M3 is doing the actual labelling, and needs every field the cascade consults.

There is also a **hard ordering constraint**: M3 cannot run first, because step 0b
(residue-aware canonicalisation) and step 2 (chemistry validation) both need
`corpus_seq` — and the corpus is M2's output. *"Is position 39 of P04075 an asparagine?"* is
unanswerable before M2 exists.

So the sequence is: **M2 parses cheaply to build the corpus → M3 re-parses fully and validates
against that corpus.** The double parse costs ~1–2 minutes on 43 MB of dbPTM files; the
alternative (holding every full parse in memory across both stages) costs far more RAM for no
benefit.

### Then, in order:

1. Build `raw_sources` (real mode: CPLM + all 37 dbPTM types + both O-GlcNAc datasets +
   N-GlycositeAtlas). **Dataset-II must be keyed with `oglcnac_dataset_ii`** — step 3 dispatches on
   that exact substring, not on anything in the data.
2. `run_m3_cached(...)`.
3. Build T1, T2, F2.
4. **OOD loop** per species: dbPTM by suffix → CPLM → qPTM (species + reliability) →
   O-GlcNAcAtlas (Dataset-I positives, Dataset-II mask). `.get()` guards absent sources.
5. T5 applicability matrix — full view and strict (homology-corrected) view side by side `[#53]`.
6. Print: discrepancy classes, agreement distributions, **full step log**, T4, T1, T2, gate.

---

# CELL 17 — Self-check `[FIX #48 hardened]`

Each assertion is a regression guard for a bug actually found:

| Assertion | Guards against |
|---|---|
| every protein has a cluster | CD-HIT parse failure |
| no cluster spans both splits | **homology leakage** |
| `labels_naive` stratum == `{ALL}` | `[#45]` naive/stratified duplication |
| every stratum ∈ `cfg.stratum_residues` | config/data drift |
| no masked site survives as a positive | step-3 leak |
| no gold negative is also a positive | `[#29]` contradiction-filter failure |
| OOD labels all carry `split == "ood"` | OOD schema drift |

Then the consolidated **`N1 RUN SUMMARY`** `[NEW #42]` — its `ood` block now also carries
`n_labelled_sites_strict` per species alongside the full count `[NEW #53]`.

---

# CELL 19 — Output contract / packaging

`save_config()`, then verifies every `REQUIRED_FILES` entry exists, lists OOD artefacts and
M_E profiles, and reports the **real recursive size** of `Paths.ROOT`.

**Measured: 52.7 MB.**

---

# OOD sources — explicitly, and how they are verified

## Which sources cover which species

**Four OOD species**, and **four contributing sources**. Two sources you might expect are absent:

| Source | Multi-species? | Used for OOD? | Why |
|---|---|---|---|
| **dbPTM** | yes — all species | **yes** | species via `entry_name` suffix (`_MOUSE`, `_ECOLI`) |
| **CPLM** | yes — per-species files | **yes** | strain-exact filenames, one download per species |
| **qPTM** | Human/Mouse/Rat/Yeast | **yes** (3 of 4) | its own `Organism` column |
| **O-GlcNAcAtlas** | 29 species | **yes** (3 of 4) | its own `species` column |
| N-GlycositeAtlas | **no** | **no** | endpoint is `download/HumanAll/` — human only |
| N-GlyDE | **no** | **no** | negative source, human benchmark only |

Per species, measured from your run:

| Species | dbPTM | CPLM | qPTM | O-GlcNAcAtlas | Final sites |
|---|---|---|---|---|---|
| mouse | ✓ | ✓ | ✓ 2.9M rows | ✓ 7,534 | **191,577** |
| rat | ✓ | ✓ | ✓ 172,871 | ✓ 741 | **59,356** |
| yeast | ✓ | ✓ | ✓ 2.3M | ✓ 501 | **89,583** |
| **E. coli** | ✓ (succinylation 4,047) | ✓ | ✗ absent | ✗ absent | **17,197** |

A `None` label in `CFG.ood_species` records a **measured absence**, not an oversight — qPTM's
`Organism` column has exactly four values and E. coli isn't one of them.

## Are OOD sites verified against UniProt? — **Yes, identically to human**

Each OOD species gets its **own Swiss-Prot reviewed proteome** from the same
`acquire_uniprot_reviewed` function, by taxonomy ID:

| Species | Taxon | Proteins (measured) |
|---|---|---|
| mouse | 10090 | 15,280 |
| rat | 10116 | 7,546 |
| yeast | 559292 | 6,229 |
| E. coli K-12 | 83333 | 4,474 |

Then `build_ood_labels` runs **the same cascade** against **that species' corpus**:

```
isoform recovery  → against the species corpus
canonicalisation  → residue looked up in the species corpus
chemistry check   → step2_validate_sites(canon, species_corpus_seq)
ambiguity mask    → species-filtered Dataset-II
dedup, pooling    → identical
```

So a mouse site is validated against the **mouse** sequence, not the human one. Species counting
(`compute_stratum_counts`) is also per species, so each OOD species has its own
*g<sub>ℓ</sub>* — background rates are directly comparable to human.

**Only two steps are skipped, and only these two:** CD-HIT and the discovery/holdout split. Both
exist to stop homologs leaking *within one corpus*; an OOD species is held out by construction.

## Homology to the human corpus — flagged, not removed `[NEW #51]`

Two OOD proteins can still be near-identical to a human one (mouse/rat especially), so a "the
latent generalised to mouse" claim risks just re-detecting a human-like pattern. `flag_human_homologs`
runs `cd-hit-2d` (human as db1, the OOD species as db2, same 40% identity as the main split) and
adds `homologous_to_human` to every OOD protein and, via `[FIX #53]`, to every OOD label row.

This is a **flag, not a filter** — both views stay available from one corpus:

| View | Question | Column |
|---|---|---|
| **Full OOD** | does the latent generalise to this species at all? | (default — no filter) |
| **Strict OOD** | does it generalise to proteins with *no* human counterpart? | `homologous_to_human == False` |

**Measured retention under strict filtering** (site-level, from the full-vs-strict table printed
in Cell 15): mouse 19.5%, rat 17.1%, yeast 86.2%, E. coli 84.9% — mammals lose most of their sites
once homologous proteins are excluded; yeast/E. coli barely change, since most of their proteins
were never homologous to human to begin with.

**This is why T5 needs both views too `[NEW #53]`.** A (species, type) pair can clear the 200-site
floor in the full view and fall below it in the strict view — mouse's `sumoylation` (212 full
sites, ~19.5% strict retention) is the concrete case. `T5_ood_applicability.csv` now reports
`testable`/`n_sites_species` (full) and `testable_strict`/`n_sites_species_strict` (strict) side
by side, so a genuine species-novelty claim can be checked against the right number instead of the
full-view count, which overstates it.

**Caveat that still stands:** the strict view controls for *whole-protein* homology, the same way
CD-HIT does for the human split — it does not rule out a shorter shared motif around the modified
residue itself. Report strict-OOD results as "no whole-protein homolog in the human corpus," not
as "sequence-novel" in an absolute sense.

---

# Is every dataset verified after the corpus is built? — Yes, and here is the order

Every source, human and OOD, passes the same two gates **against the corpus**:

| Gate | Question | Failure bucket |
|---|---|---|
| **Membership** | is this accession in the corpus? | `accession_not_in_corpus` |
| **Chemistry** | is the residue at this position right for this type? | `residue_mismatch` / `position_out_of_range` |

**Isoforms specifically** — the answer to "are isoforms verified?" is **yes, twice**:

1. `strip_isoform_suffix` only rewrites `P12345-2` → `P12345` **if the stripped form exists in the
   corpus** (`stripped.isin(corpus_seq.keys())`). It never invents an accession.
2. The rewritten row then goes through the **normal** chemistry check like any other row.

So a recovered isoform site is verified exactly as strictly as a directly-matched one.

**The residual risk, restated precisely:** positions in isoform records refer to the *isoform's*
numbering. If isoform-2 has an insertion near the N-terminus, position 100 there is position 90 in
the canonical sequence. The chemistry check catches most such shifts (a shifted position usually
lands on the wrong amino acid) — but a shift can coincidentally land on a *valid* residue and
slip through. **This is why every recovered row carries `from_isoform_accession=True`:** re-run
any analysis with those rows excluded and confirm nothing material changes.

Measured recovery: **14,868 CPLM rows** and **15,765 dbPTM Phosphorylation rows**.

---

# File reference — what each output contains and who consumes it

## Corpus vs. labels — the distinction that matters most

These are the two central files, and confusing them is the easiest way to misread the pipeline.

**One sentence each:**
- `corpus.parquet` = **one row per PROTEIN.** "Here is a protein, its sequence, and which half of
  the split it belongs to."
- `labels_stratified.parquet` = **one row per (SITE, PTM TYPE).** "Here is a *specific residue*
  inside a protein that carries a *specific* modification."

### Side by side

| | `corpus.parquet` | `labels_stratified.parquet` |
|---|---|---|
| Unit of a row | **one protein** | **one (position, PTM type) pair** |
| Rows (measured) | **18,128** | **498,619** |
| Built by | M2 | M3 |
| Built from | Swiss-Prot sequences | PTM database coordinates |
| Holds sequences? | **yes — the only file that does** | no, just `residue` (one letter) |
| Holds PTM info? | no | yes |
| Same protein appears | exactly once | **many times** — once per modified site |
| Answers | "what is this protein, and is it discovery or holdout?" | "which residues are modified, with what, and how confidently?" |

### Sample rows — the same protein in both files

**`corpus.parquet`** — P10000 appears **exactly once**:

| accession | length | sequence | cluster_id | split |
|---|---|---|---|---|
| `P10000` | `412` | `MKSTYRNQAALP...` *(412 chars)* | `164` | `holdout` |

**`labels_stratified.parquet`** — the same protein appears **once per annotated site**:

| accession | position | residue | type | stratum | source_count | crosstalk | cluster_id | split |
|---|---|---|---|---|---|---|---|---|
| `P10000` | `68` | `N` | `N-glycosylation` | `N` | 1 | False | `164` | `holdout` |
| `P10000` | `101` | `N` | `N-glycosylation` | `N` | 2 | False | `164` | `holdout` |
| `P10000` | `205` | `K` | `succinylation` | `K` | 1 | **True** | `164` | `holdout` |
| `P10000` | `205` | `K` | `acetylation` | `K` | 2 | **True** | `164` | `holdout` |

Read the last two rows carefully: **position 205 appears twice**, because that one lysine carries
*two different* modifications. That is exactly what `crosstalk=True` records — and it's why
succinylation's crosstalk fraction is 90.7%.

### Full label schema

| Column | Meaning | Origin |
|---|---|---|
| `accession` | UniProt accession | source DB |
| `position` | 1-based residue index | source DB |
| **`residue`** | the actual amino acid there | **looked up from the corpus** `[NEW #33]` |
| `type` | canonical PTM type | step 0b |
| `stratum` | which background population applies | `cfg.ptm_type_to_stratum` |
| `type_pooled` | `type`, or `<stratum>::other_PTM` if <200 sites | step 5 |
| `headline_eligible` | ≥1,000 sites | step 5 |
| `source_count` | independent sources reporting it (**max 2**) | step 4 |
| `crosstalk` | this residue also carries another type | `crosstalk_flags` |
| `cluster_id`, `split` | **inherited from the corpus** | step 6 |
| `from_isoform_accession` | recovered by stripping `-2` | step 0a `[NEW #27]` |
| `evidence`, `lp`, `lp_class`, `pmid` | carried through from source | source DB |

### Why they must be joined, not merged

`cluster_id` and `split` appear in **both** files — that is the join key, deliberately duplicated
so label files are self-contained.

Downstream you need both, for different halves of the same question:

$$\text{enrichment} = \frac{\text{how often a latent fires on } \textbf{labelled sites}}{\text{how often it fires on } \textbf{all residues of that stratum}}$$

- The **numerator** comes from `labels_stratified.parquet` — 13,981 N-glycosylation sites.
- The **denominator** comes from `corpus.parquet` via `stratum_counts.json` — all **263,374** N
  residues in the corpus.

The labels file alone cannot give you the denominator: it only contains residues that *are*
modified. **With positives only, every rate is 1.0 and enrichment is identically 1.** That is why
the corpus is not optional bookkeeping — it is half the statistic.

> **This is also the answer to "where are the negatives?"** They are not stored. A negative is
> "a residue of the right type that is *not* in the labels file and *not* in the exclusion mask."
> It's computed as a set difference at query time:
> `negatives = (all residues of stratum, from corpus) − (positives) − (masked)`.

## Core corpus artefacts (M2)

| File | Contents | Consumed by |
|---|---|---|
| **`corpus.parquet`** | `accession, sequence, length, cluster_id, split` — one row per protein. **The single source of truth for sequences.** | N2 (forward passes), N3 (background), N4 |
| **`stratum_counts.json`** | `{K: 425120, ST: 983282, ..., total_residues: 7474423}` — exact *g<sub>ℓ</sub>*. | **N3 — every enrichment denominator** |
| **`split_manifest.json`** | cluster→split map, seed, gate report. Reproducibility record. | audit / N3 V12b |

## Label sets (M3)

| File | Contents | Consumed by |
|---|---|---|
| **`labels_stratified.parquet`** | **the main label set.** Positives only: `accession, position, residue, stratum, type, type_pooled, headline_eligible, source_count, crosstalk, cluster_id, split, from_isoform_accession` | N3 (M7 enrichment), N4 |
| **`labels_naive.parquet`** | same sites, `stratum="ALL"` → whole-corpus background. **Sensitivity comparison against stratified.** | N3 |
| **`labels_qptm.parquet`** | same schema, labels from qPTM only. **CF-26 replication.** | N3 |
| **`labels_ood_<species>.parquet`** ×4 | same schema, `split="ood"`, `+species` | N4 cross-species |

## Masks and negatives

| File | Contents | Consumed by |
|---|---|---|
| **`exclusion_mask.parquet`** | `accession, position, type, mask_reason` — 6,382 rows. Sites in **neither** positives nor negatives. | N3 — **must be subtracted from every background** |
| **`gold_negatives_nglyco.parquet`** | 1,088 verified non-glycosylated sequons | N3 Tier-1 validation |
| **`gold_negatives_contradicted.parquet`** | the 222 removed — keeps the 16.95% rate reportable | thesis methods |
| **`crosstalk.parquet`** | residues carrying >1 type | N3 confounding analysis |

## Study sets

| File | Contents | Consumed by |
|---|---|---|
| **`site_agreement.parquet`** | per site: sources reporting, sources covering **(protein, type)**, disagreement count | N4 adjudication |
| **`annotation_discrepancies.parquet`** | 3 disagreement classes + `question_for_latent` | **N4 — the adjudication study** |
| **`qptm_dbptm_overlap.csv`** | per-type overlap — **required alongside any CF-26 claim** | thesis |

## Tables and provenance

| File | Contents |
|---|---|
| `manifest.json` | SHA + size + timestamp per download |
| `step_log.json` | per-source per-step counts for the whole cascade |
| `config.json` | full config dump — exact reproducibility |
| `T1` | provenance: per source × step |
| `T2` | label composition: type × stratum, **incl. crosstalk_fraction** |
| `T3` | split composition: **achieved** holdout fraction per type |
| `T4` | per-type survival waterfall |
| `T5` | OOD applicability (CF-22) — full view **and** strict (no-human-homolog) view `[#53]` |
| `TE_*` ×8 | M_E diagnostics |
| `F2` | dataset composition figure, panels (a)–(e) |
| `exploration/*.json` | per-source profiles |

> **The three N3 must not miss:** `stratum_counts.json` (denominators),
> `exclusion_mask.parquet` (must be removed from backgrounds), and the positives-only convention
> below.

---

# Data-flow summary

```
UniProt ─┐
CPLM  ───┤
dbPTM ───┼─► M1 ─► resolved{} ─► M_E (diagnose, change nothing)
qPTM  ───┤            │
Atlases ─┘            ├─► M2: filter → g_ell → CD-HIT 40% → stratified split
                      │        └─► corpus.parquet, stratum_counts.json, T3
                      │        └─► OOD ×4 (no CD-HIT, no split)
                      └─► M3: isoform → canonicalise → evidence → chemistry
                               → mask → dedup → frequency → inherit split
                               ├─► labels_stratified / labels_naive
                               ├─► labels_qptm       (parallel, CF-26)
                               ├─► labels_ood_×4     (parallel, no split)
                               ├─► exclusion_mask, site_agreement,
                               │   annotation_discrepancies, gold negatives
                               └─► T1, T2, T4, T5, F2
```

---

# Where the plan and the code differ

| Plan says | Code does | Why |
|---|---|---|
| `labels_*` carry `label`, `negative_tier`, `corpus` | **positives only** | Same as the original. Negatives derived at query time in Phase V. Materialising every (residue, type) pair = tens of millions of rows. **N3 must know this.** |
| CD-HIT at 50% | **40%** | Your instruction |
| Step 1 discards by-similarity records | runs, **drops nothing** | Every source used is already experimentally filtered at origin |
| Step 3b masks CPLM LP II/III | **inert** | `lp` is 100% null in the real download |
| F2 panels (a)–(e) | all five `[FIX #48]` | (d) was previously missing |
| six steps | **eight** (0a, 0b added) | isoform recovery + residue-aware canonicalisation |
| 3 sources drive stratification | **44 types, all sources** | `[FIX #40]` |

---

# Open item for N3

`labels_*.parquet` store **positives only**. The negative population is reconstructed from
`corpus.parquet` + `stratum_counts.json` + `exclusion_mask.parquet` + `gold_negatives_nglyco.parquet`
+ `accessions_carrying_type()`. This matches the original pipeline's behaviour but **not** the
plan's stated schema. N3 must be built against the actual convention.
