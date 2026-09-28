"""Top-level orchestrator: M1 (acquisition) -> M2 (clustering) -> M3 (label cascade).

Runnable via `python -m ptm_sae.corpus.pipeline [--mock] [--force]`. Ported from N1.ipynb's own
cell sequence (cells 6, 9, 12, 15, 17); M_E (cell 8, exploratory diagnostics) is deliberately not
ported -- it is optional tooling, not part of the output contract.

Emits, under `CorpusPaths.processed_dir`: corpus.parquet, labels_stratified.parquet,
stratum_counts.json, split_manifest.json, exclusion_mask.parquet, gold_negatives_nglyco.parquet.

`--mock` runs the whole pipeline against small synthetic fixtures (no network, no `cd-hit`
binary required for the *acquisition* half, but `cd-hit` itself must still be on PATH for M2 --
mock mode only replaces the raw data, not CD-HIT). Use it for local smoke-testing this module;
the real, full-scale build runs exclusively on Kaggle (a later stage, not part of this port).
"""

from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from ptm_sae.corpus import acquisition, clustering, labels
from ptm_sae.corpus.acquisition import CPLM_SPECIES_HUMAN
from ptm_sae.corpus.config import CFG, CorpusPaths
from ptm_sae.data.splitting import subdivide_discovery_clusters


def _build_m2_inputs(resolved: dict, mock: bool) -> tuple[dict, dict]:
    """M2's raw-type-hit table uses M1's actual resolved paths, never a hard-coded path. Feeds
    every positive-bearing source, not a three-source subset, so the stratified split balances
    on the full label population it needs to balance."""
    m1_raw_paths = {"cplm_human": resolved["cplm"][CPLM_SPECIES_HUMAN]}
    parsers = {
        "cplm_human": lambda p: acquisition.parse_cplm_zip(p)[
            ["accession", "position", "type"]
        ]
    }

    if mock:
        return m1_raw_paths, parsers

    for t, p in resolved.get("dbptm", {}).items():
        key = f"dbptm_{t.replace(' ', '_').replace('/', '_')}"
        m1_raw_paths[key] = p
        parsers[key] = lambda p, t=t: acquisition.parse_dbptm_flatfile(p, t)[
            ["accession", "position", "type"]
        ]

    if "Dataset-I" in resolved.get("oglcnac_atlas", {}):
        m1_raw_paths["oglcnac_dataset_i"] = resolved["oglcnac_atlas"]["Dataset-I"]
        parsers["oglcnac_dataset_i"] = lambda p: (
            acquisition.parse_oglcnac_atlas_flatfile(p)[
                ["accession", "position", "type"]
            ]
        )

    if "glycosite_atlas" in resolved:
        m1_raw_paths["glycosite_atlas"] = resolved["glycosite_atlas"]
        parsers["glycosite_atlas"] = lambda p: (
            acquisition.parse_nglycosite_atlas_flatfile(p)[
                ["accession", "position", "type"]
            ]
        )

    return m1_raw_paths, parsers


def _build_m3_raw_sources(
    resolved: dict, corpus: pd.DataFrame, mock: bool
) -> tuple[dict, pd.DataFrame | None]:
    """M3's full-column raw_sources dict (accession/position/type/evidence/pmid), distinct from
    M2's lightweight positional-only join above."""
    cplm_raw = acquisition.parse_cplm_zip(resolved["cplm"][CPLM_SPECIES_HUMAN])
    cplm_raw = cplm_raw.copy()
    cplm_raw["evidence"] = (
        "MS/MS"  # CPLM's compact schema has no per-site evidence field
    )
    raw_sources = {
        "cplm_human": cplm_raw[["accession", "position", "type", "evidence"]]
    }

    if mock:
        # Synthetic N-glycosylation source for a real, if tiny, cascade run without network.
        rng = random.Random(7)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
        rows = []
        accs = corpus["accession"].tolist()
        for i in range(300):
            acc = rng.choice(accs)
            seq = corpus.loc[corpus["accession"] == acc, "sequence"].iloc[0]
            n_positions = [j + 1 for j, ch in enumerate(seq) if ch == "N"]
            if not n_positions:
                continue
            pos = (
                rng.choice(n_positions)
                if rng.random() > 0.1
                else rng.randint(1, len(seq))
            )
            rows.append(
                {
                    "accession": acc,
                    "position": pos,
                    "type": "N-linked Glycosylation",
                    "evidence": "MS/MS",
                    "pmid": str(40000000 + i),
                }
            )
        raw_sources["dbptm_nglyco_MOCK"] = pd.DataFrame(rows)
        return raw_sources, None  # no Gold-tier negatives fixture in mock mode

    for dbptm_type, dbptm_path in resolved.get("dbptm", {}).items():
        key = f"dbptm_{dbptm_type.replace(' ', '_').replace('/', '_')}"
        raw_sources[key] = acquisition.parse_dbptm_flatfile(dbptm_path, dbptm_type)

    oglcnac = resolved.get("oglcnac_atlas", {})
    if "Dataset-I" in oglcnac:
        raw_sources["oglcnac_dataset_i"] = acquisition.parse_oglcnac_atlas_flatfile(
            oglcnac["Dataset-I"]
        )
    if "Dataset-II" in oglcnac:
        raw_sources["oglcnac_dataset_ii"] = acquisition.parse_oglcnac_atlas_flatfile(
            oglcnac["Dataset-II"]
        )

    if "glycosite_atlas" in resolved:
        raw_sources["glycosite_atlas"] = acquisition.parse_nglycosite_atlas_flatfile(
            resolved["glycosite_atlas"]
        )

    glyde_30pct = None
    if "glyde" in resolved:
        glyde_parts = [
            acquisition.parse_nglyde_flatfile(p) for p in resolved["glyde"].values()
        ]
        glyde_30pct = pd.concat(glyde_parts, ignore_index=True) if glyde_parts else None

    return raw_sources, glyde_30pct


def _self_check(corpus: pd.DataFrame, result: dict) -> None:
    """Structural regression guards -- ported from N1's cell 17, trimmed to what this
    subpackage's reduced output set can actually check (no labels_naive/OOD assertions)."""
    assert corpus["cluster_id"].notna().all()
    assert set(corpus["split"].unique()) <= {"discovery", "holdout"}
    per_cluster_splits = corpus.groupby("cluster_id")["split"].nunique()
    assert (per_cluster_splits == 1).all(), (
        "a CD-HIT cluster was split across discovery/holdout -- leakage bug"
    )
    assert result["labels_stratified"]["split"].isin(["discovery", "holdout"]).all()
    assert not result["labels_stratified"]["type"].isna().any(), (
        "unmapped PTM type slipped through canonicalisation"
    )
    assert set(result["labels_stratified"]["stratum"].unique()) <= set(
        CFG.stratum_residues
    ), "a stratum appeared that is not in cfg.stratum_residues"

    mask_keys = (
        set(
            zip(
                result["exclusion_mask"]["accession"].astype(str),
                result["exclusion_mask"]["position"].astype(int),
                result["exclusion_mask"]["type"],
                strict=False,
            )
        )
        if len(result["exclusion_mask"])
        else set()
    )
    pos_keys = set(
        zip(
            result["labels_stratified"]["accession"].astype(str),
            result["labels_stratified"]["position"].astype(int),
            result["labels_stratified"]["type"],
            strict=False,
        )
    )
    assert not (mask_keys & pos_keys), (
        "a masked site survived as a positive -- step 3 leak"
    )

    if len(result.get("gold_negatives_n_glycosylation", [])):
        gold = set(
            zip(
                result["gold_negatives_n_glycosylation"]["accession"].astype(str),
                result["gold_negatives_n_glycosylation"]["position"].astype(int),
                strict=False,
            )
        )
        n_glyco = result["labels_stratified"][
            result["labels_stratified"]["type"] == "N-glycosylation"
        ]
        pos_n = set(
            zip(
                n_glyco["accession"].astype(str),
                n_glyco["position"].astype(int),
                strict=False,
            )
        )
        assert not (gold & pos_n), (
            "a gold-tier negative is also an annotated positive -- contradiction filter failed"
        )

    print("N1 structural self-check: OK")


def _finalize_output_schema(
    corpus: pd.DataFrame, result: dict, paths: CorpusPaths
) -> tuple[pd.DataFrame, dict]:
    """Renames N1's native `accession`/`split`/`type` columns to the canonical `uniprot_id`/
    `partition`/`ptm_type` schema `training/dataset.py` and `training/collapse_check.py` consume,
    and subdivides N1's two-way `discovery`/`holdout` split into the three-way
    `discovery_train`/`discovery_val`/`held_out` partition the training loop needs -- reusing
    `subdivide_discovery_clusters` (already built for exactly this) rather than re-deriving a
    split from scratch. Overwrites corpus.parquet/labels_stratified.parquet/exclusion_mask.parquet/
    gold_negatives_nglyco.parquet under `paths.processed_dir` with the finalized schema."""
    cluster_tokens = corpus.groupby("cluster_id")["length"].sum().to_dict()
    cluster_profiles = {
        cid: {"tokens": tokens} for cid, tokens in cluster_tokens.items()
    }
    discovery_cluster_ids = (
        corpus.loc[corpus["split"] == "discovery", "cluster_id"].unique().tolist()
    )
    subdivision = subdivide_discovery_clusters(
        cluster_profiles, discovery_cluster_ids, train_ratio=0.875
    )

    def _partition_for(row: pd.Series) -> str:
        if row["split"] == "holdout":
            return "held_out"
        return subdivision.get(row["cluster_id"], "discovery_train")

    corpus = corpus.copy()
    corpus["partition"] = corpus.apply(_partition_for, axis=1)
    corpus = corpus.drop(columns=["split"]).rename(columns={"accession": "uniprot_id"})
    uid_to_partition = dict(
        zip(corpus["uniprot_id"], corpus["partition"], strict=False)
    )

    labels_stratified = (
        result["labels_stratified"]
        .drop(columns=["split"])
        .rename(columns={"accession": "uniprot_id", "type": "ptm_type"})
    )
    labels_stratified["partition"] = labels_stratified["uniprot_id"].map(
        uid_to_partition
    )

    exclusion_mask = result["exclusion_mask"].rename(
        columns={"accession": "uniprot_id"}
    )
    gold_negatives = result["gold_negatives_n_glycosylation"].rename(
        columns={"accession": "uniprot_id"}
    )

    corpus.to_parquet(paths.corpus_parquet, index=False)
    labels_stratified.to_parquet(paths.labels_stratified, index=False)
    exclusion_mask.to_parquet(paths.exclusion_mask, index=False)
    if len(gold_negatives.columns):
        gold_negatives.to_parquet(paths.gold_negatives_nglyco, index=False)

    result = {
        **result,
        "labels_stratified": labels_stratified,
        "exclusion_mask": exclusion_mask,
        "gold_negatives_n_glycosylation": gold_negatives,
    }
    return corpus, result


def run(mock: bool = False, force: bool = False) -> dict:
    paths = CorpusPaths.from_env()

    manifest, resolved = acquisition.run_m1(paths, mock=mock, force=force)
    print(f"M1: {len(manifest.entries)} artefacts -> {paths.manifest}")

    m1_raw_paths, parsers = _build_m2_inputs(resolved, mock=mock)
    corpus, gate_report, m2_cache_info = clustering.run_m2(
        resolved["swissprot"], m1_raw_paths, parsers, CFG, paths, force=force
    )
    stratum_counts = json.loads(paths.stratum_counts.read_text())
    print(
        f"M2: {len(corpus)} proteins, {corpus['cluster_id'].nunique()} clusters "
        f"({'CACHED' if m2_cache_info['cached'] else 'built fresh'}), "
        f"split counts: {corpus['split'].value_counts().to_dict()}"
    )

    raw_sources, glyde_30pct = _build_m3_raw_sources(resolved, corpus, mock=mock)
    result, m3_cache_info = labels.run_m3_cached(
        raw_sources,
        corpus,
        CFG,
        paths,
        glyde_30pct=glyde_30pct,
        corpus_fingerprint=m2_cache_info["fingerprint"],
        force=force,
    )
    print(
        f"M3: labels_stratified={len(result['labels_stratified'])} rows "
        f"({'CACHED' if m3_cache_info['cached'] else 'built fresh'}), "
        f"exclusion_mask={len(result['exclusion_mask'])} rows"
    )

    _self_check(corpus, result)

    discovery_headline_ok = {
        t: bool(
            result["labels_stratified"][
                (result["labels_stratified"]["type"] == t)
                & (result["labels_stratified"]["split"] == "discovery")
            ].shape[0]
            >= CFG.min_sites_headline
        )
        for t in CFG.primary_types
    }
    print(
        f"Gate -- primary types clearing the >=1000-site headline bar IN DISCOVERY: {discovery_headline_ok}"
    )
    if not all(discovery_headline_ok.values()) and not mock:
        print(
            "ACTION REQUIRED: at least one primary type falls below 1,000 discovery-partition "
            "sites. Reduce CFG.holdout_fraction to 0.15 and re-run M2+M3 before proceeding."
        )
    if mock:
        print(
            "(mock mode: the headline gate is expected to fail on this tiny synthetic corpus.)"
        )

    corpus, result = _finalize_output_schema(corpus, result, paths)
    print(f"Finalized partitions: {corpus['partition'].value_counts().to_dict()}")

    print(
        f"\nRequired artifacts written under {paths.processed_dir}: corpus.parquet, "
        f"labels_stratified.parquet, stratum_counts.json, split_manifest.json, "
        f"exclusion_mask.parquet, gold_negatives_nglyco.parquet"
    )
    return {
        "corpus": corpus,
        "gate_report": gate_report,
        "stratum_counts": stratum_counts,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the N1-derived PTM corpus (M1 -> M2 -> M3)."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use small synthetic fixtures instead of real network sources.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore every cache; rebuild from scratch."
    )
    args = parser.parse_args()
    run(mock=args.mock, force=args.force)


if __name__ == "__main__":
    main()
