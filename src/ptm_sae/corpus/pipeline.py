"""Top-level orchestrator: M1 (acquisition) -> M2 (clustering) -> M3 (label cascade).

Runnable via `python -m ptm_sae.corpus.pipeline [--mock] [--force] [--resplit]`. Ported from N1.ipynb's own
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
import dataclasses
import json
import random

import pandas as pd

from ptm_sae import runtime
from ptm_sae.corpus import acquisition, clustering, labels
from ptm_sae.corpus.acquisition import CPLM_SPECIES_HUMAN
from ptm_sae.corpus.config import CFG, CorpusPaths
from ptm_sae.data.splitting import PARTITIONS, SplitSettings
from ptm_sae.extraction.hub import HfSyncClient

# verify_outputs tolerances. The splitter alone reaches ~0.1% mean / ~1% max, but the homology audit
# merges hundreds of clusters into a few giant indivisible ones (measured on the real corpus: 423
# merges; held-out then holds only 10.5% of the tokens in clusters >10 proteins instead of 20%, mean
# deviation 2.2%, max 48%). The gate therefore only has to catch what matters: a regression to the
# old heuristic (31% mean / 100% max), with the per-type site floors and exact token windows enforced
# separately by the splitter.
MAX_MEAN_DEVIATION = 0.03
MAX_FEATURE_DEVIATION = 0.6
# Residual cross-boundary homologs tolerated per audited partition. 1% of proteins biases a held-out
# estimate by well under its sampling noise (for 200+ sites, about 7% relative), while the unmerged
# CD-HIT@40% split leaked ~10%. cd-hit-2d is itself a heuristic, so "zero" would be false precision.
MAX_LEAK_FRACTION = 0.01

ARTIFACT_FILENAMES = [
    "corpus.parquet",
    "labels_stratified.parquet",
    "stratum_counts.json",
    "split_manifest.json",
    "exclusion_mask.parquet",
    "gold_negatives_nglyco.parquet",
]


def upload_to_hf(paths: CorpusPaths, repo_id: str, subpath: str) -> None:
    hub = HfSyncClient()
    for filename in ARTIFACT_FILENAMES:
        file_path = paths.processed_dir / filename
        if not file_path.exists():
            print(f"  [HF] {filename}: not produced this run, skipping")
            continue
        result = hub.upload_shard(repo_id, subpath, file_path)
        print(f"  [HF] {filename}: {'uploaded' if result else 'already up to date'}")


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


def _split_settings(mock: bool = False, audit: bool = True) -> SplitSettings:
    """CFG.split, with the held-out site floor tied to the corpus-wide `holdout_min_sites_per_type`.
    Mock mode relaxes every hard constraint: the synthetic corpus is far too small for them."""
    settings = dataclasses.replace(
        CFG.split,
        held_min_sites=CFG.holdout_min_sites_per_type,
        homology_audit=audit and CFG.split.homology_audit,
    )
    if not mock:
        return settings
    return dataclasses.replace(
        settings,
        token_tolerance=1.0,
        held_min_sites=0,
        val_min_sites=0,
        min_type_sites=1,
        max_cluster_share=1.1,
    )


def _apply_split(
    corpus: pd.DataFrame, sites: pd.DataFrame, settings: SplitSettings
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Assigns the canonical `partition` (discovery_train / discovery_val / held_out) to the
    corpus and its labels via the joint split in `clustering.assign_partitions`. Expects the
    canonical `uniprot_id` schema. Clusters may come back merged by the homology audit."""
    cluster_of, partition_of_cluster, report = clustering.assign_partitions(
        corpus, sites, CFG.stratum_residues, settings, CFG.cdhit_identity
    )
    corpus = corpus.assign(cluster_id=corpus["uniprot_id"].map(cluster_of))
    corpus["partition"] = corpus["cluster_id"].map(partition_of_cluster)
    sites = sites.assign(cluster_id=sites["uniprot_id"].map(cluster_of))
    sites["partition"] = sites["cluster_id"].map(partition_of_cluster)

    assert set(corpus["partition"]) <= set(PARTITIONS), "protein without a partition"
    assert not sites["partition"].isna().any(), "labelled site without a partition"
    assert (corpus.groupby("cluster_id")["partition"].nunique() == 1).all(), (
        "a homology cluster was split across partitions -- leakage bug"
    )
    report["cluster_to_partition"] = {int(c): p for c, p in partition_of_cluster.items()}
    return corpus, sites, report


def _write_split_outputs(
    paths: CorpusPaths, corpus: pd.DataFrame, sites: pd.DataFrame, report: dict
) -> None:
    """Writes corpus.parquet / labels_stratified.parquet and adds the split provenance to
    split_manifest.json next to the N1 keys M2 already wrote (`cluster_to_split` is N1's own
    discovery/holdout assignment, kept for reproducibility)."""
    corpus.to_parquet(paths.corpus_parquet, index=False)
    sites.to_parquet(paths.labels_stratified, index=False)
    manifest = (
        json.loads(paths.split_manifest.read_text())
        if paths.split_manifest.exists()
        else {}
    )
    manifest.update(
        {
            key: report[key]
            for key in ("split_version", "balance", "homology_audit", "cluster_to_partition")
        }
    )
    paths.split_manifest.write_text(json.dumps(manifest, indent=2, default=str))


def _finalize_output_schema(
    corpus: pd.DataFrame, result: dict, paths: CorpusPaths, mock: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Renames N1's native `accession`/`split`/`type` columns to the canonical `uniprot_id`/
    `partition`/`ptm_type` schema `training/dataset.py` and `training/collapse_check.py` consume,
    and replaces N1's two-way `discovery`/`holdout` split with the joint three-way
    `discovery_train`/`discovery_val`/`held_out` partition (data/splitting.py). N1's own
    assignment is kept as the `split_n1` column. Overwrites corpus.parquet/
    labels_stratified.parquet/exclusion_mask.parquet/gold_negatives_nglyco.parquet under
    `paths.processed_dir` with the finalized schema."""
    corpus = corpus.rename(columns={"accession": "uniprot_id", "split": "split_n1"})
    sites = (
        result["labels_stratified"]
        .drop(columns=["split"])
        .rename(columns={"accession": "uniprot_id", "type": "ptm_type"})
    )
    corpus, sites, split_report = _apply_split(corpus, sites, _split_settings(mock))

    exclusion_mask = result["exclusion_mask"].rename(columns={"accession": "uniprot_id"})
    gold_negatives = result["gold_negatives_n_glycosylation"].rename(
        columns={"accession": "uniprot_id"}
    )

    _write_split_outputs(paths, corpus, sites, split_report)
    exclusion_mask.to_parquet(paths.exclusion_mask, index=False)
    if len(gold_negatives.columns):
        gold_negatives.to_parquet(paths.gold_negatives_nglyco, index=False)

    result = {
        **result,
        "labels_stratified": sites,
        "exclusion_mask": exclusion_mask,
        "gold_negatives_n_glycosylation": gold_negatives,
    }
    return corpus, result


def verify_outputs(paths: CorpusPaths | None = None, mock: bool = False) -> list[str]:
    """Checks the finalized corpus against everything a publish must be able to promise and
    returns the problems found (empty = safe to publish): all three partitions present, every
    homology cluster inside one partition, the final homology audit clean, per-feature balance
    within tolerance, and each primary PTM type above the headline bar in discovery. Mock builds
    only get the structural checks, since the synthetic corpus cannot meet the real thresholds."""
    paths = paths or CorpusPaths.from_env()
    problems: list[str] = []
    corpus = pd.read_parquet(paths.corpus_parquet)
    sites = pd.read_parquet(paths.labels_stratified)
    manifest = json.loads(paths.split_manifest.read_text())

    # 1. Structure
    present = set(corpus["partition"].dropna())
    if present != set(PARTITIONS):
        problems.append(f"partitions present {sorted(present)}, expected {list(PARTITIONS)}")
    if corpus["partition"].isna().any():
        problems.append("proteins without a partition")
    if (corpus.groupby("cluster_id")["partition"].nunique() > 1).any():
        problems.append("a homology cluster spans several partitions")
    if sites["partition"].isna().any():
        problems.append("labelled sites without a partition")
    if mock:
        return problems

    # 2. Homology audit: the last pass must have run, with the residual leak fraction within tolerance
    final_pass = (manifest.get("homology_audit") or [{}])[-1]
    if final_pass.get("skipped"):
        problems.append("homology audit was skipped")
    elif "leaks" in final_pass:
        for name, leaks in final_pass["leaks"].items():
            fraction = leaks / max(final_pass["queries"][name], 1)
            if fraction > MAX_LEAK_FRACTION:
                problems.append(f"{leaks} of {final_pass['queries'][name]} {name} proteins ({fraction:.2%}) still have a homolog across the boundary (tolerance {MAX_LEAK_FRACTION:.1%})")
    elif final_pass.get("cross_partition_pairs", 1) > 0:  # manifests written before per-boundary counts
        problems.append(f"{final_pass.get('cross_partition_pairs')} cross-partition pairs remain after the last audit pass")

    # 3. Balance
    for part, stats in manifest["balance"]["summary"].items():
        if stats["mean_relative_deviation"] > MAX_MEAN_DEVIATION or stats["max_relative_deviation"] > MAX_FEATURE_DEVIATION:
            problems.append(f"{part} balance off target: {stats}")

    # 4. Headline bar per primary type, in discovery
    in_discovery = sites[sites["partition"] != "held_out"]
    for ptm_type in CFG.primary_types:
        n_sites = int((in_discovery["ptm_type"] == ptm_type).sum())
        if n_sites < CFG.min_sites_headline:
            problems.append(f"{ptm_type}: {n_sites} discovery sites < headline bar {CFG.min_sites_headline}")
    return problems


def resplit(paths: CorpusPaths | None = None, audit: bool = True) -> dict:
    """Re-runs only the final three-way split on an already-built corpus in
    `paths.processed_dir`: no downloads, no CD-HIT clustering, no label cascade -- the clusters
    and labels are already in corpus.parquet / labels_stratified.parquet. `audit=False` skips the
    cd-hit-2d homology audit (recorded in the manifest) for machines without CD-HIT."""
    paths = paths or CorpusPaths.from_env()
    corpus = pd.read_parquet(paths.corpus_parquet)
    sites = pd.read_parquet(paths.labels_stratified)
    if "split_n1" not in corpus.columns:  # built before the joint split: N1 held out == held_out
        corpus["split_n1"] = (corpus["partition"] == "held_out").map(
            {True: "holdout", False: "discovery"}
        )
    corpus, sites, report = _apply_split(
        corpus.drop(columns=["partition"]),
        sites.drop(columns=["partition"]),
        _split_settings(audit=audit),
    )
    _write_split_outputs(paths, corpus, sites, report)

    print(f"Partitions: {corpus['partition'].value_counts().to_dict()}")
    print(f"Balance (mean/max relative deviation): {report['balance']['summary']}")
    print(f"Homology audit: {report['homology_audit']}")
    return report


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

    corpus, result = _finalize_output_schema(corpus, result, paths, mock=mock)
    print(f"Finalized partitions: {corpus['partition'].value_counts().to_dict()}")

    final_labels = result["labels_stratified"]
    discovery_headline_ok = {
        t: bool(
            ((final_labels["ptm_type"] == t) & (final_labels["partition"] != "held_out")).sum()
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
            "sites. Lower the held-out token share in CFG.split.fractions (e.g. to 0.15) and "
            "re-run with --resplit before proceeding."
        )
    if mock:
        print(
            "(mock mode: the headline gate is expected to fail on this tiny synthetic corpus.)"
        )

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
    runtime.ensure_utf8_output()
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
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Only re-run the final train/val/held-out split on the existing corpus.parquet and "
        "labels_stratified.parquet (no downloads, no CD-HIT clustering).",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="With --resplit: skip the cd-hit-2d homology audit (for machines without CD-HIT).",
    )
    args = parser.parse_args()
    if args.resplit:
        resplit(audit=not args.skip_audit)
        return
    run(mock=args.mock, force=args.force)


if __name__ == "__main__":
    main()
