"""Maintenance of an already-built corpus: `verify_outputs` (publish gates), `resplit` (the joint
three-way partition) and `upload_to_hf`. Runnable as `python -m ptm_sae.corpus.pipeline [--skip-audit]`,
which re-splits the corpus in `CorpusPaths.processed_dir`.

The build itself (M1 acquisition, M2 clustering, M3 label cascade, the N1 two-way split and the
`--mock` smoke mode) was archived: the code lives on the git branch `archive/corpus-build`
(restore with `git checkout archive/corpus-build -- src/ptm_sae/corpus/`), and the published corpus is
the Hub dataset `mustafa-muhaimin/ptm-sae-corpus`. Emits/consumes, under `CorpusPaths.processed_dir`:
corpus.parquet, labels_stratified.parquet, stratum_counts.json, split_manifest.json,
exclusion_mask.parquet, gold_negatives_nglyco.parquet.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from ptm_sae import runtime
from ptm_sae.corpus import clustering
from ptm_sae.corpus.config import CFG, CorpusPaths
from ptm_sae.data.splitting import PARTITIONS, SplitSettings
from ptm_sae.extraction.hub import HfSyncClient

# verify_outputs tolerances. The splitter alone reaches ~0.1% mean / ~1% max, but the homology audit
# merges hundreds of clusters into a few giant indivisible ones, which moves whole large families to
# one side: on the real corpus the share of tokens in clusters >10 proteins ends 59-65% off in val and
# held-out. That drift is a consequence of homology closure, not a split defect, so the cluster-size
# bins (`size_*`) stay in the manifest as reported numbers and the gate covers the other features
# (residue and PTM-type composition: ~2% mean in val, ~0.6% in held-out, max 17%). It only has to catch
# what matters: a regression to the old heuristic (31% mean / 100% max), with the per-type site floors
# and exact token windows enforced separately by the splitter.
MAX_MEAN_DEVIATION = 0.03
MAX_FEATURE_DEVIATION = 0.6
REPORTED_ONLY_FEATURE_PREFIX = "size_"
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


def _split_settings(audit: bool = True) -> SplitSettings:
    """CFG.split, with the cd-hit-2d homology audit optionally switched off."""
    return dataclasses.replace(CFG.split, homology_audit=audit and CFG.split.homology_audit)


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
    split_manifest.json, keeping any keys an earlier build wrote there."""
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


def verify_outputs(paths: CorpusPaths | None = None) -> list[str]:
    """Checks the finalized corpus against everything a publish must be able to promise and
    returns the problems found (empty = safe to publish): all three partitions present, every
    homology cluster inside one partition, the final homology audit clean, per-feature balance
    within tolerance, and each primary PTM type above the headline bar in discovery."""
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
    gated = {
        name: feature["relative_deviation"]
        for name, feature in manifest["balance"]["features"].items()
        if not name.startswith(REPORTED_ONLY_FEATURE_PREFIX)
    }
    for part in manifest["balance"]["summary"]:
        deviations = [by_part[part] for by_part in gated.values()]
        stats = {
            "mean_relative_deviation": sum(deviations) / len(deviations),
            "max_relative_deviation": max(deviations),
            "features_over_25pct": sum(d > 0.25 for d in deviations),
        }
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


def main() -> None:
    runtime.ensure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Re-run the final train/val/held-out split on the corpus in CorpusPaths.processed_dir "
        "(no downloads, no CD-HIT clustering)."
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the cd-hit-2d homology audit (for machines without CD-HIT).",
    )
    resplit(audit=not parser.parse_args().skip_audit)


if __name__ == "__main__":
    main()
