"""M3 -- Label construction: the filtering cascade. Ported from N1.ipynb cells 14-15.

Inputs: M1 databases (raw site tables), M2 corpus (sequences, cluster_id, split).
Emits: labels_stratified.parquet, exclusion_mask.parquet, gold_negatives_nglyco.parquet.
Gate: N-glycosylation, O-GlcNAcylation, succinylation must each clear >=1000 headline-bar sites.

Six steps, each recording counts, executed in order:
  1. Evidence tier      -- keep MS/MS + low-throughput experimental; drop "by similarity" etc.
  2. Site-mapping        -- residue at (accession,position) must match the type's chemistry
  3. Ambiguity masking    -- O-GlcNAcAtlas Dataset-II (binary) + O-glycosylation/O-GlcNAc containment
  4. Cross-source dedup   -- merge on (accession,position,type); source_count = n sources agreeing
  5. Minimum frequency    -- <200/stratum -> pooled to "other PTM"; >=1000 -> headline-eligible
  6. Redundancy           -- inherit cluster_id/split from M2 (cluster-level, not protein-level)

This module intentionally does NOT port N1's `labels_naive` variant, the qPTM independent-
replication sub-pipeline (`build_qptm_labels`), OOD-species labels, or the T1/T2/T4/F2/
site_agreement/annotation_discrepancy diagnostic tables -- none of those are in the artifact
contract this subpackage targets (`corpus.parquet`, `labels_stratified.parquet`,
`stratum_counts.json`, `split_manifest.json`, `exclusion_mask.parquet`,
`gold_negatives_nglyco.parquet`). The six-step cascade itself, isoform recovery, chemistry
validation, and the Gold-tier negative contradiction filter are ported faithfully.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from ptm_sae.corpus.config import (
    CFG,
    PTM_TYPE_RESIDUE_RESOLVED,
    CorpusPaths,
    _normalize_type_key,
    config_fingerprint,
    find_cached,
    normalize_ptm_type,
)

# Chemistry ground truth: which residue identity each PTM type can occur on. Derived from
# cfg.ptm_type_to_stratum rather than hand-maintained, so every type in that dict is
# automatically covered here too.
PTM_VALID_RESIDUES: dict[str, set] = {
    t: CFG.stratum_residues[s]
    for t, s in CFG.ptm_type_to_stratum.items()
    if s in CFG.stratum_residues
}

EVIDENCE_KEEP = {"ms/ms", "low-throughput", "experimental", "mass spectrometry"}
EVIDENCE_DROP = {"by similarity", "potential", "probable", "text-mining"}


def canonicalize_types(
    df: pd.DataFrame, cfg, corpus_seq: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Residue-aware canonicalisation. When `corpus_seq` is supplied, the real amino acid at each
    site is looked up FIRST and passed to `normalize_ptm_type`, so multi-chemistry source labels
    ("Phosphorylation", "Methylation", qPTM's bare "Glycosylation") resolve to the right sibling
    type instead of being collapsed onto one and then silently dropped by step2."""
    df = df.copy()
    known = set(cfg.ptm_type_to_stratum)

    if corpus_seq is not None:
        residues = []
        for acc, pos in zip(df["accession"].astype(str), df["position"], strict=False):
            seq = corpus_seq.get(acc)
            if seq is None or pd.isna(pos):
                residues.append(None)
                continue
            p = int(pos)
            residues.append(seq[p - 1] if 1 <= p <= len(seq) else None)
        # Memoised on (raw_type, residue) -- the distinct (type, residue) space is tiny, so this
        # avoids re-scanning ~50 canonical keys per row on multi-million-row sources.
        _resolve_cache: dict[tuple, str] = {}
        resolved_types = []
        for t, r in zip(df["type"], residues, strict=False):
            key = (t, r)
            hit = _resolve_cache.get(key)
            if hit is None:
                hit = normalize_ptm_type(t, known_types=known, residue=r)
                _resolve_cache[key] = hit
            resolved_types.append(hit)
        df["type"] = resolved_types
    else:
        df["type"] = df["type"].map(lambda t: normalize_ptm_type(t, known_types=known))

    # Two DIFFERENT kinds of unmapped row, which must not be conflated:
    #   (a) residue-unresolved -- the raw label IS in PTM_TYPE_RESIDUE_RESOLVED, but this row's
    #       residue could not be determined. Expected and benign: DROPPED here, counted.
    #   (b) genuinely unknown spelling -- no table knows this label at all. Still RAISES.
    unknown_mask = ~df["type"].isin(known)
    residue_resolved_labels = set(PTM_TYPE_RESIDUE_RESOLVED)
    raw_keys = (
        df["type"].map(_normalize_type_key)
        if not unknown_mask.any()
        else pd.Series([_normalize_type_key(t) for t in df["type"]], index=df.index)
    )
    resolvable_mask = unknown_mask & raw_keys.isin(residue_resolved_labels)
    n_residue_unresolved = int(resolvable_mask.sum())
    if n_residue_unresolved:
        df = df[~resolvable_mask]
        unknown_mask = unknown_mask[~resolvable_mask]

    unknown_types = sorted(df.loc[unknown_mask, "type"].unique().tolist())
    return df, {
        "n_rows": len(df),
        "n_unmapped_rows": int(unknown_mask.sum()),
        "n_residue_unresolved_dropped": n_residue_unresolved,
        "unmapped_types": unknown_types,
        "residue_aware": corpus_seq is not None,
    }


def step1_evidence_filter(
    df: pd.DataFrame, evidence_col: str = "evidence"
) -> tuple[pd.DataFrame, dict]:
    if evidence_col not in df.columns:
        # No per-site evidence field (e.g. CPLM's curated events) -> already filtered at the
        # database level, retained as-is.
        return df.copy(), {
            "n_in": len(df),
            "n_dropped": 0,
            "n_out": len(df),
            "reason": "no evidence field",
        }
    ev = df[evidence_col].astype(str).str.lower().str.strip()
    keep_mask = ev.apply(
        lambda s: (
            any(k in s for k in EVIDENCE_KEEP)
            and not any(d in s for d in EVIDENCE_DROP)
        )
    )
    out = df[
        keep_mask
    ].copy()  # unrecognised evidence strings default to DROP (conservative)
    return out, {
        "n_in": len(df),
        "n_dropped": int((~keep_mask).sum()),
        "n_out": len(out),
    }


def step2_validate_sites(
    df: pd.DataFrame,
    corpus_seq: dict[str, str],
    valid_residues: dict[str, set] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Residue at (accession, position) [1-indexed, UniProt convention] must match the type's
    known chemistry -- catches off-by-one / stale-isoform errors that are otherwise silent."""
    from tqdm.auto import tqdm

    valid_residues = (
        valid_residues if valid_residues is not None else PTM_VALID_RESIDUES
    )
    accs = df["accession"].values
    poss = df["position"].values
    types = df["type"].values
    keep = []
    fail_reasons = defaultdict(int)
    for acc, pos, t in tqdm(
        zip(accs, poss, types, strict=False),
        total=len(accs),
        desc="[M3] Site validation",
        unit="site",
        leave=False,
    ):
        seq = corpus_seq.get(acc)
        if seq is None:
            keep.append(False)
            fail_reasons["accession_not_in_corpus"] += 1
            continue
        pos = int(pos)
        if not (1 <= pos <= len(seq)):
            keep.append(False)
            fail_reasons["position_out_of_range"] += 1
            continue
        residue = seq[pos - 1]
        valid = valid_residues.get(t)
        if valid is None:
            keep.append(True)  # unknown type -> chemistry unchecked, not rejected here
            continue
        if residue not in valid:
            keep.append(False)
            fail_reasons["residue_mismatch"] += 1
            continue
        keep.append(True)
    out = df[pd.Series(keep, index=df.index)].copy()
    return out, {
        "n_in": len(df),
        "n_dropped": len(df) - len(out),
        "n_out": len(out),
        "failure_breakdown": dict(fail_reasons),
    }


def step3a_oglcnac_mask(dataset_ii: pd.DataFrame) -> pd.DataFrame:
    """3a: every candidate S/T in a Dataset-II (ambiguous) peptide window -> exclusion mask row."""
    rows = []
    for _, r in dataset_ii.iterrows():
        rows.append(
            {
                "accession": r["accession"],
                "position": int(r["position"]),
                "type": "O-GlcNAcylation",
                "mask_reason": "oglcnac_atlas_dataset_ii_ambiguous",
            }
        )
    return pd.DataFrame(rows)


def step3c_oglycosylation_ambiguity_mask(
    positives: pd.DataFrame, oglcnac_sites: set
) -> pd.DataFrame:
    """O-GlcNAcylation is a SUBTYPE of O-glycosylation, not a sibling -- dbPTM's broad "O-linked
    Glycosylation" bucket contains both mucin-type O-glycosylation AND O-GlcNAc. Sites labelled
    O-glycosylation but absent from O-GlcNAcAtlas are masked for O-GlcNAcylation ONLY (their own
    O-glycosylation label is untouched); an unknown share of them may be genuinely uncatalogued
    O-GlcNAc."""
    o_glyco = positives[positives["type"] == "O-glycosylation"]
    if not len(o_glyco):
        return pd.DataFrame(columns=["accession", "position", "type", "mask_reason"])
    keys = list(
        zip(o_glyco["accession"].astype(str), o_glyco["position"], strict=False)
    )
    rows = [
        {
            "accession": a,
            "position": int(p),
            "type": "O-GlcNAcylation",
            "mask_reason": "dbptm_o_linked_glyco_may_be_uncatalogued_o_glcnac",
        }
        for a, p in keys
        if (a, int(p)) not in oglcnac_sites
    ]
    return pd.DataFrame(rows, columns=["accession", "position", "type", "mask_reason"])


def step4_dedup(dfs: list[pd.DataFrame], source_labels=None) -> pd.DataFrame:
    """Merge on (accession, position, type). Emits BOTH `source_count` (distinct sources
    reporting the site) and `source_row_count` (contributing rows -- a proxy for how heavily a
    site is attested WITHIN a source). `source_labels` gives each row's source: one label per
    FRAME (len == len(dfs)) or one per ROW (len == total rows); if omitted, each frame in `dfs`
    is treated as one source."""
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    key_cols = ["accession", "position", "type"]

    if source_labels is None:
        src = np.repeat(np.arange(len(dfs)), [len(d) for d in dfs])
    elif len(source_labels) == len(dfs) and len(dfs) != len(combined):
        src = np.repeat(np.asarray(source_labels, dtype=object), [len(d) for d in dfs])
    elif len(source_labels) == len(combined):
        src = np.asarray(source_labels, dtype=object)
    else:
        raise ValueError(
            f"source_labels has length {len(source_labels)}; expected {len(dfs)} (one per frame) "
            f"or {len(combined)} (one per row)"
        )

    combined = combined.copy()
    combined["_src"] = src
    g = combined.groupby(key_cols)
    grouped = g["_src"].nunique().reset_index(name="source_count")
    grouped["source_row_count"] = g.size().reset_index(name="_n")["_n"].values
    combined = combined.drop(columns="_src")
    aux_cols = [
        c for c in combined.columns if c not in key_cols
    ]  # first non-null pmid/evidence
    first_vals = combined.groupby(key_cols)[aux_cols].first().reset_index()
    out = grouped.merge(first_vals, on=key_cols, how="left")
    return out


def step5_min_frequency(
    df: pd.DataFrame, cfg, group_by_stratum: bool, type_to_stratum: dict[str, str]
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["stratum"] = df["type"].map(type_to_stratum)
    if group_by_stratum:
        counts = df.groupby(["stratum", "type"]).size().reset_index(name="n")
    else:
        counts = df.groupby(["type"]).size().reset_index(name="n")
        counts["stratum"] = "ALL"
        df["stratum"] = (
            "ALL"  # what actually makes the naive variant naive (see N1_CHANGELOG #45)
        )

    rare = counts[counts["n"] < cfg.min_sites_per_type]
    headline = set(counts.loc[counts["n"] >= cfg.min_sites_headline, "type"])

    rare_types = set(rare["type"])
    df["type_pooled"] = df["type"].where(
        ~df["type"].isin(rare_types), df["stratum"] + "::other_PTM"
    )
    df["headline_eligible"] = df["type"].isin(headline)

    report = {
        "n_types_before_pooling": int(counts["type"].nunique()),
        "n_types_pooled_to_other": len(rare_types),
        "pooled_types": sorted(rare_types),
        "headline_eligible_types": sorted(headline),
    }
    return df, report


def step6_inherit_split(df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    lookup = corpus.set_index("accession")[["cluster_id", "split"]]
    out = df.join(lookup, on="accession")
    n_missing = out["split"].isna().sum()
    if n_missing:
        raise RuntimeError(
            f"{n_missing} labelled sites have no corpus split assignment (accession not in M2 corpus)"
        )
    return out


def strip_isoform_suffix(
    df: pd.DataFrame, corpus_seq: dict[str, str], accession_col: str = "accession"
) -> tuple[pd.DataFrame, dict]:
    """UniProt isoform accessions (P12345-2) never match the canonical corpus. Stripping the
    suffix recovers rows whose base accession IS in the corpus.

    CAVEAT, deliberately not hidden: the reported position refers to the ISOFORM's numbering, so
    stripping makes it a canonical-sequence position that may be wrong wherever the isoform
    differs. step2's chemistry check is the safety net, but a shifted position CAN coincidentally
    land on a valid residue -- every recovered row is flagged `from_isoform_accession=True`."""
    df = df.copy()
    accs = df[accession_col].astype(str)
    is_isoform = (
        accs.str.contains("-") & accs.str.rsplit("-", n=1).str[-1].str.isdigit()
    )
    stripped = accs.str.rsplit("-", n=1).str[0]
    recoverable = (
        is_isoform & ~accs.isin(corpus_seq.keys()) & stripped.isin(corpus_seq.keys())
    )
    df["from_isoform_accession"] = recoverable
    df.loc[recoverable, accession_col] = stripped[recoverable]
    return df, {
        "n_rows": len(df),
        "n_isoform_rows": int(is_isoform.sum()),
        "n_rows_recovered_by_stripping": int(recoverable.sum()),
    }


def crosstalk_flags(
    positive_sites: pd.DataFrame,
    source_frame: pd.DataFrame | None = None,
    source_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Mark every (accession, position) carrying >1 type; record the type combination. When
    `source_frame` + `source_labels` are supplied, also classifies whether one source asserted
    the co-modification itself (`same_source_multi_type`) or it was assembled from different
    sources (`cross_source_only`, which does not by itself prove disagreement)."""
    grp = positive_sites.groupby(["accession", "position"])["type"].apply(
        lambda s: sorted(set(s))
    )
    crosstalk = grp[grp.apply(len) > 1].reset_index()
    crosstalk = crosstalk.rename(columns={"type": "types"})
    crosstalk["n_types"] = crosstalk["types"].apply(len)
    crosstalk["type_combination"] = crosstalk["types"].apply(lambda ts: " + ".join(ts))
    crosstalk["crosstalk"] = True

    if source_frame is None or source_labels is None or not len(source_frame):
        crosstalk["sources_per_type"] = None
        crosstalk["evidence_class"] = "unattributed"
        return crosstalk

    sf = source_frame.copy()
    sf["_source"] = list(source_labels)
    per_site: dict[tuple, dict[str, set]] = {}
    for acc, pos, typ, src in zip(
        sf["accession"].astype(str),
        sf["position"],
        sf["type"],
        sf["_source"],
        strict=False,
    ):
        if pd.isna(pos):
            continue
        per_site.setdefault((acc, int(pos)), {}).setdefault(typ, set()).add(src)

    sources_per_type, evidence_class = [], []
    for acc, pos in zip(
        crosstalk["accession"].astype(str), crosstalk["position"], strict=False
    ):
        tmap = per_site.get((acc, int(pos)), {})
        sources_per_type.append(
            "; ".join(f"{t}={','.join(sorted(v))}" for t, v in sorted(tmap.items()))
        )
        src_counts: dict[str, int] = {}
        for srcs in tmap.values():
            for sv in srcs:
                src_counts[sv] = src_counts.get(sv, 0) + 1
        evidence_class.append(
            "same_source_multi_type"
            if any(v > 1 for v in src_counts.values())
            else "cross_source_only"
        )
    crosstalk["sources_per_type"] = sources_per_type
    crosstalk["evidence_class"] = evidence_class
    return crosstalk


def accessions_carrying_type(positive_sites: pd.DataFrame, ptm_type: str) -> set:
    """Defines the 'Hard' negative tier: proteins with >=1 annotated site of this type. Consumed
    downstream (training/collapse_check.py's query-time negative derivation, a later stage)."""
    return set(positive_sites.loc[positive_sites["type"] == ptm_type, "accession"])


def build_gold_negatives_nglyde(
    glyde_30pct: pd.DataFrame, positive_sites: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The N-GlyDE non-glycosylated sequons -- the only Gold-tier negatives in the project.
    Removes sites CONTRADICTED by an annotated N-glycosylation positive elsewhere. Returns
    (clean_negatives, contradicted_negatives, report); the contradicted set is kept, not
    discarded, so the contamination rate stays reportable."""
    neg = glyde_30pct[glyde_30pct["Label"].str.lower().str.contains("non")].copy()
    neg = neg.rename(columns={"UniProt Accession": "accession", "Position": "position"})
    neg["type"] = "N-glycosylation"
    neg["tier"] = "gold"
    neg = neg[["accession", "position", "type", "tier"]]

    if positive_sites is None or not len(positive_sites):
        return (
            neg,
            neg.iloc[0:0].copy(),
            {"n_gold": len(neg), "n_contradicted": 0, "contradiction_check_run": False},
        )
    pos_n = positive_sites[positive_sites["type"] == "N-glycosylation"]
    pos_keys = set(
        zip(pos_n["accession"].astype(str), pos_n["position"].astype(int), strict=False)
    )
    contradicted_mask = [
        (str(a), int(p)) in pos_keys
        for a, p in zip(neg["accession"], neg["position"], strict=False)
    ]
    contradicted = neg[pd.Series(contradicted_mask, index=neg.index)]
    clean = neg[~pd.Series(contradicted_mask, index=neg.index)]
    report = {
        "n_gold_before": len(neg),
        "n_contradicted": len(contradicted),
        "n_gold_clean": len(clean),
        "pct_contradicted": round(100 * len(contradicted) / len(neg), 2)
        if len(neg)
        else 0.0,
        "contradiction_check_run": True,
    }
    if len(contradicted):
        print(
            f"[M3] Gold-tier negatives: removed {len(contradicted)}/{len(neg)} "
            f"({report['pct_contradicted']}%) contradicted by an annotated N-glycosylation "
            f"positive. {len(clean)} clean gold negatives remain."
        )
    return clean, contradicted, report


def _enrich_label_schema(
    labels: pd.DataFrame,
    corpus_seq: dict[str, str],
    crosstalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adds the per-site columns the implementation plan's labels_stratified schema specifies:
    `residue` (the actual amino acid) and `crosstalk` (this residue also carries another PTM
    type). NOT added here, deliberately: `label`, `negative_tier`, `corpus` -- those only make
    sense on a materialised positive+negative row set, and these files store POSITIVES ONLY. The
    negative population is reconstructed downstream (a later stage) from corpus.parquet +
    stratum_counts.json + exclusion_mask.parquet + accessions_carrying_type (Hard tier) +
    gold_negatives_nglyco.parquet (Gold tier)."""
    out = labels.copy()
    if "residue" not in out.columns:
        out["residue"] = [
            corpus_seq[a][int(p) - 1]
            if a in corpus_seq and 1 <= int(p) <= len(corpus_seq[a])
            else None
            for a, p in zip(out["accession"].astype(str), out["position"], strict=False)
        ]
    if crosstalk is not None and len(crosstalk):
        ct_keys = set(
            zip(
                crosstalk["accession"].astype(str),
                crosstalk["position"].astype(int),
                strict=False,
            )
        )
        out["crosstalk"] = [
            (str(a), int(p)) in ct_keys
            for a, p in zip(out["accession"], out["position"], strict=False)
        ]
    else:
        out["crosstalk"] = False
    return out


def run_m3(
    raw_sources: dict[str, pd.DataFrame],
    corpus: pd.DataFrame,
    cfg,
    glyde_30pct: pd.DataFrame | None = None,
) -> dict:
    """raw_sources: dict of label -> raw DataFrame, each with at least [accession, position,
    type] and optionally [evidence, pmid]. This is the cascade itself (always recomputes) -- see
    run_m3_cached for the cache-aware entry point."""
    corpus_seq = dict(zip(corpus["accession"], corpus["sequence"], strict=False))
    step_log = {}
    masks = []

    canonicalized = {}
    for label, df in raw_sources.items():
        # Isoform-suffix recovery runs BEFORE canonicalisation, so residue-aware type resolution
        # sees an accession that actually resolves against the corpus.
        stripped, log_iso = strip_isoform_suffix(df, corpus_seq)
        if log_iso["n_rows_recovered_by_stripping"]:
            step_log[f"step0_isoform_recovery::{label}"] = log_iso
        out0, log0 = canonicalize_types(stripped, cfg, corpus_seq=corpus_seq)
        step_log[f"step0_canonicalize::{label}"] = log0
        if log0["unmapped_types"]:
            raise ValueError(
                f"[{label}] {log0['n_unmapped_rows']} rows have PTM type spellings not in "
                f"PTM_TYPE_ALIASES or cfg.ptm_type_to_stratum: {log0['unmapped_types']}. "
                f"Add them to PTM_TYPE_ALIASES rather than letting them silently vanish."
            )
        canonicalized[label] = out0

    after_step1 = {}
    for label, df in canonicalized.items():
        out1, log1 = step1_evidence_filter(df)
        step_log[f"step1::{label}"] = log1
        after_step1[label] = out1

    after_step2 = {}
    for label, df in after_step1.items():
        out2, log2 = step2_validate_sites(df, corpus_seq)
        step_log[f"step2::{label}"] = log2
        after_step2[label] = out2

    positives_for_dedup, positive_source_labels = [], []
    for label, df in after_step2.items():
        if "oglcnac_dataset_ii" in label:
            masks.append(step3a_oglcnac_mask(df))
            continue  # Dataset-II contributes only to the mask, never to positives
        positives_for_dedup.append(df)
        positive_source_labels.append(label)

    if positives_for_dedup:
        all_positives = pd.concat(positives_for_dedup, ignore_index=True, sort=False)
        oglcnac_confirmed = set(
            zip(
                all_positives.loc[
                    all_positives["type"] == "O-GlcNAcylation", "accession"
                ].astype(str),
                all_positives.loc[
                    all_positives["type"] == "O-GlcNAcylation", "position"
                ].astype(int),
                strict=False,
            )
        )
        oglyco_mask = step3c_oglycosylation_ambiguity_mask(
            all_positives, oglcnac_confirmed
        )
        if len(oglyco_mask):
            masks.append(oglyco_mask)
        step_log["step3c_oglcnac_containment"] = {
            "n_oglcnac_confirmed_sites": len(oglcnac_confirmed),
            "n_oglyco_sites_masked_for_oglcnac": len(oglyco_mask),
        }

    exclusion_mask = (
        pd.concat(masks, ignore_index=True, sort=False)
        if masks
        else pd.DataFrame(columns=["accession", "position", "type", "mask_reason"])
    )

    mask_keys = set(
        zip(
            exclusion_mask["accession"],
            exclusion_mask["position"],
            exclusion_mask["type"],
            strict=False,
        )
    )
    filtered_positives = []
    for df in positives_for_dedup:
        keep = ~df.apply(
            lambda r: (r["accession"], r["position"], r["type"]) in mask_keys, axis=1
        )
        filtered_positives.append(df[keep])

    deduped = step4_dedup(filtered_positives, source_labels=positive_source_labels)
    step_log["step4::dedup"] = {
        "n_in": sum(len(d) for d in filtered_positives),
        "n_out": len(deduped),
    }

    labels_stratified, report_strat = step5_min_frequency(
        deduped, cfg, group_by_stratum=True, type_to_stratum=cfg.ptm_type_to_stratum
    )
    step_log["step5::stratified"] = report_strat

    labels_stratified = step6_inherit_split(labels_stratified, corpus)
    step_log["step6::split_inherited"] = {"n_rows": len(labels_stratified)}

    source_labels = []
    for label, df in zip(positive_source_labels, filtered_positives, strict=False):
        source_labels.extend([label] * len(df))
    agreement_input = (
        pd.concat(filtered_positives, ignore_index=True, sort=False)
        if filtered_positives
        else pd.DataFrame()
    )

    ct = crosstalk_flags(
        deduped,
        source_frame=agreement_input,
        source_labels=pd.Series(source_labels)
        if len(source_labels) == len(agreement_input)
        else None,
    )
    labels_stratified = _enrich_label_schema(labels_stratified, corpus_seq, ct)

    if glyde_30pct is not None:
        gold_negatives, gold_contradicted, gold_report = build_gold_negatives_nglyde(
            glyde_30pct, positive_sites=deduped
        )
        step_log["gold_negatives::contradiction_filter"] = gold_report
    else:
        gold_negatives = pd.DataFrame()
        gold_contradicted = pd.DataFrame()

    return {
        "labels_stratified": labels_stratified,
        "exclusion_mask": exclusion_mask,
        "crosstalk": ct,
        "gold_negatives_n_glycosylation": gold_negatives,
        "gold_negatives_contradicted": gold_contradicted,
        "step_log": step_log,
    }


def _dataframe_fingerprint(df: pd.DataFrame | None) -> str:
    """Content hash of a DataFrame's actual values (not just shape/columns), so the cache
    invalidates on real data changes."""
    if df is None or len(df) == 0:
        return "empty"
    try:
        h = pd.util.hash_pandas_object(df, index=True).values
        import hashlib as _hashlib

        return _hashlib.sha256(h.tobytes()).hexdigest()[:16]
    except (TypeError, ValueError):
        return json.dumps(
            {"shape": list(df.shape), "columns": list(df.columns)}, default=str
        )


def run_m3_cached(
    raw_sources: dict[str, pd.DataFrame],
    corpus: pd.DataFrame,
    cfg,
    paths: CorpusPaths,
    glyde_30pct: pd.DataFrame | None = None,
    corpus_fingerprint: str | None = None,
    force: bool = False,
) -> tuple[dict, dict]:
    """Cache-aware entry point. Emits labels_stratified/exclusion_mask/gold_negatives (parquet,
    under paths.processed_dir) + step_log.json, either by copying a valid cache hit or by
    running the cascade and writing fresh output."""
    sources_fp = {
        label: _dataframe_fingerprint(df) for label, df in sorted(raw_sources.items())
    }
    corpus_fp = (
        corpus_fingerprint
        if corpus_fingerprint is not None
        else _dataframe_fingerprint(corpus)
    )
    expected_fp = config_fingerprint(
        {
            "corpus_fingerprint": corpus_fp,
            "raw_sources_fingerprint": sources_fp,
            "glyde_fingerprint": _dataframe_fingerprint(glyde_30pct),
        },
        cfg=cfg,
    )

    companion_files = {
        "labels_stratified": "labels_stratified.parquet",
        "exclusion_mask": "exclusion_mask.parquet",
        "crosstalk": "crosstalk.parquet",
        "gold_negatives_n_glycosylation": "gold_negatives_nglyco.parquet",
        "gold_negatives_contradicted": "gold_negatives_contradicted.parquet",
    }

    if not force:
        hit = find_cached(
            companion_files["labels_stratified"], search_dirs=[paths.processed_dir]
        )
        if hit is not None:
            fp_sidecar = hit.with_suffix(hit.suffix + ".fp")
            step_log_src = hit.parent / "step_log.json"
            if (
                fp_sidecar.exists()
                and fp_sidecar.read_text().strip() == expected_fp
                and step_log_src.exists()
            ):
                src_dir = hit.parent
                result = {}
                for key, fname in companion_files.items():
                    src = src_dir / fname
                    dest = paths.processed_dir / fname
                    if src != dest and src.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                    result[key] = (
                        pd.read_parquet(dest) if dest.exists() else pd.DataFrame()
                    )
                result["step_log"] = json.loads(step_log_src.read_text())
                print(
                    f"[M3] CACHE HIT: reused labels_stratified/exclusion_mask/gold_negatives, "
                    f"skipped the six-step cascade on {sum(len(d) for d in raw_sources.values())} raw rows."
                )
                return result, {"cached": True, "fingerprint": expected_fp}
            print(
                "[M3] found cached labels but fingerprint is stale -- rebuilding the six-step cascade."
            )

    print("[M3] no valid cache -- running the six-step filtering cascade.")
    result = run_m3(raw_sources, corpus, cfg, glyde_30pct=glyde_30pct)
    for key, fname in companion_files.items():
        dest = paths.processed_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        result[key].to_parquet(dest, index=False)
    (paths.processed_dir / companion_files["labels_stratified"]).with_suffix(
        ".parquet.fp"
    ).write_text(expected_fp)
    (paths.processed_dir / "step_log.json").write_text(
        json.dumps(result["step_log"], indent=2, default=str)
    )
    return result, {"cached": False, "fingerprint": expected_fp}
