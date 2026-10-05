"""Homology audit, cluster merging and the corpus-level split entry points (offline: cd-hit faked)."""

import dataclasses
import itertools
import json
import subprocess

import pandas as pd
import pytest

from ptm_sae.corpus import clustering, pipeline
from ptm_sae.corpus.config import CFG, CorpusPaths
from ptm_sae.data.splitting import PARTITIONS

STRATUM_RESIDUES = {"K": {"K"}, "ST": {"S", "T"}}


def test_merge_leaking_clusters_is_transitive_and_keeps_the_smallest_id():
    cluster_of = pd.Series({"a": 5, "b": 9, "c": 12, "d": 3}, name="cluster_id")

    merged, absorbed = clustering.merge_leaking_clusters(cluster_of, [("a", "b"), ("b", "c")])

    assert merged.to_dict() == {"a": 5, "b": 5, "c": 5, "d": 3}
    assert absorbed == 2


def test_merge_leaking_clusters_without_pairs_changes_nothing():
    cluster_of = pd.Series({"a": 1, "b": 2})

    merged, absorbed = clustering.merge_leaking_clusters(cluster_of, [])

    assert merged.to_dict() == {"a": 1, "b": 2}
    assert absorbed == 0


def _fake_cdhit_2d(clstr_by_call: list[str]):
    """subprocess.run stand-in: writes the given .clstr text for each successive cd-hit-2d call."""
    calls = iter(clstr_by_call)

    def run(cmd, text=True):
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path + ".clstr", "w") as f:
            f.write(next(calls))
        return subprocess.CompletedProcess(cmd, 0)

    return run


def test_find_cross_partition_pairs_parses_clstr_and_filters_short_sequences(
    monkeypatch, tmp_path
):
    proteins = pd.DataFrame(
        {
            "uniprot_id": ["t1", "t2", "v1", "h1", "tiny"],
            "sequence": ["ACDEFGHIK", "LMNPQRSTV", "ACDEFGHIL", "WWWYYYKKK", "A"],
            "partition": [
                "discovery_train",
                "discovery_train",
                "discovery_val",
                "held_out",
                "held_out",
            ],
        }
    )
    # Boundary 1 (held_out vs discovery): h1 reaches t1. Boundary 2 (val vs train): v1 reaches t1.
    boundary_1 = ">Cluster 0\n0\t9aa, >t1... *\n1\t9aa, >h1... at 45.00%\n>Cluster 1\n0\t9aa, >t2... *\n"
    boundary_2 = ">Cluster 0\n0\t9aa, >t1... *\n1\t9aa, >v1... at 88.89%\n"
    monkeypatch.setattr(clustering, "check_cdhit_available", lambda: None)
    monkeypatch.setattr(subprocess, "run", _fake_cdhit_2d([boundary_1, boundary_2]))

    pairs = clustering.find_cross_partition_pairs(proteins, identity=0.4, tmp_dir=tmp_path)

    assert pairs == [("t1", "h1"), ("t1", "v1")]
    # Sequences shorter than the word size never reach cd-hit; each boundary got its own FASTAs.
    query_fasta = (tmp_path / "audit0_query.fasta").read_text()
    assert ">h1" in query_fasta and ">tiny" not in query_fasta
    assert ">t1" in (tmp_path / "audit0_reference.fasta").read_text()
    assert ">v1" in (tmp_path / "audit1_query.fasta").read_text()


def test_assign_partitions_merges_leaking_clusters_and_resolves(
    monkeypatch, make_corpus, small_split_settings
):
    proteins, sites = make_corpus(n_clusters=120)
    calls = []

    def fake_audit(frame, identity, tmp_dir):
        calls.append(identity)
        if len(calls) > 1:
            return []
        held = frame.loc[frame["partition"] == "held_out", "uniprot_id"].iloc[0]
        train = frame.loc[frame["partition"] == "discovery_train", "uniprot_id"].iloc[0]
        return [(train, held)]

    monkeypatch.setattr(clustering, "find_cross_partition_pairs", fake_audit)

    cluster_of, partition_of_cluster, report = clustering.assign_partitions(
        proteins, sites, STRATUM_RESIDUES, small_split_settings
    )

    assert [entry["cross_partition_pairs"] for entry in report["homology_audit"]] == [1, 0]
    assert report["homology_audit"][0]["clusters_merged"] == 1
    # After the merge every cluster still lives in exactly one partition.
    partition_of_protein = proteins["uniprot_id"].map(cluster_of).map(partition_of_cluster)
    assert set(partition_of_protein) <= set(PARTITIONS)
    assert cluster_of.nunique() == proteins["cluster_id"].nunique() - 1


def test_assign_partitions_can_skip_the_audit(monkeypatch, make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=100)

    def must_not_run(*args, **kwargs):
        raise AssertionError("audit ran although homology_audit=False")

    monkeypatch.setattr(clustering, "find_cross_partition_pairs", must_not_run)

    _, _, report = clustering.assign_partitions(
        proteins, sites, STRATUM_RESIDUES, dataclasses.replace(small_split_settings, homology_audit=False)
    )

    assert report["homology_audit"] == [{"pass": 0, "skipped": True}]


def test_resplit_rewrites_outputs_and_keeps_n1_provenance(
    monkeypatch, tmp_path, make_corpus, small_split_settings
):
    proteins, sites = make_corpus(n_clusters=120)
    paths = CorpusPaths(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
    # A corpus as published before the joint split: old-style partition, N1 keys in the manifest.
    proteins["partition"] = ["held_out" if c % 5 == 0 else "discovery_train" for c in proteins["cluster_id"]]
    sites["partition"] = sites["uniprot_id"].map(dict(zip(proteins["uniprot_id"], proteins["partition"], strict=True)))
    sites["ptm_type"] = sites["type_pooled"]
    proteins.to_parquet(paths.corpus_parquet, index=False)
    sites.to_parquet(paths.labels_stratified, index=False)
    paths.split_manifest.write_text(json.dumps({"cluster_to_split": {"0": "holdout"}, "seed": 42}))
    monkeypatch.setattr(
        pipeline,
        "CFG",
        dataclasses.replace(CFG, split=small_split_settings, stratum_residues=STRATUM_RESIDUES),
    )

    report = pipeline.resplit(paths, audit=False)

    corpus = pd.read_parquet(paths.corpus_parquet)
    labels = pd.read_parquet(paths.labels_stratified)
    assert set(corpus["partition"]) == set(PARTITIONS)
    assert set(corpus.loc[corpus["cluster_id"] % 5 == 0, "split_n1"]) == {"holdout"}
    assert (corpus.groupby("cluster_id")["partition"].nunique() == 1).all()
    assert labels["partition"].notna().all()
    manifest = json.loads(paths.split_manifest.read_text())
    assert manifest["cluster_to_split"] == {"0": "holdout"} and manifest["seed"] == 42
    assert manifest["split_version"] == report["split_version"]
    assert manifest["homology_audit"] == [{"pass": 0, "skipped": True}]
    assert len(manifest["cluster_to_partition"]) == corpus["cluster_id"].nunique()

    # A second re-split starts from the already-finalized schema (split_n1 present) and still works.
    pipeline.resplit(paths, audit=False)
    assert pd.read_parquet(paths.corpus_parquet)["split_n1"].isin(["holdout", "discovery"]).all()


def test_split_settings_relax_hard_constraints_only_in_mock_mode():
    strict = pipeline._split_settings()
    mock = pipeline._split_settings(mock=True)

    assert strict.held_min_sites == CFG.holdout_min_sites_per_type
    assert strict.token_tolerance == CFG.split.token_tolerance
    assert mock.token_tolerance == 1.0 and mock.held_min_sites == 0


def test_audit_requires_the_cdhit_binary(monkeypatch):
    monkeypatch.setattr(clustering.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="cd-hit"):
        clustering.find_cross_partition_pairs(
            pd.DataFrame({"uniprot_id": ["a"], "sequence": ["ACDEF"], "partition": ["held_out"]})
        )


def _finalized_corpus(monkeypatch, tmp_path, make_corpus, small_split_settings):
    """A corpus finalized by the real resplit code path (audit skipped), plus its CorpusPaths."""
    proteins, sites = make_corpus(n_clusters=120)
    paths = CorpusPaths(tmp_path / "raw", tmp_path / "interim", tmp_path / "processed")
    proteins["partition"] = "discovery_train"
    sites["partition"] = "discovery_train"
    sites["ptm_type"] = sites["type_pooled"]
    proteins.to_parquet(paths.corpus_parquet, index=False)
    sites.to_parquet(paths.labels_stratified, index=False)
    monkeypatch.setattr(
        pipeline,
        "CFG",
        dataclasses.replace(
            CFG,
            split=small_split_settings,
            stratum_residues=STRATUM_RESIDUES,
            primary_types=("phospho",),
            min_sites_headline=10,
        ),
    )
    pipeline.resplit(paths, audit=False)
    return paths


def test_verify_outputs_mock_mode_checks_structure_only(
    monkeypatch, tmp_path, make_corpus, small_split_settings
):
    paths = _finalized_corpus(monkeypatch, tmp_path, make_corpus, small_split_settings)

    assert pipeline.verify_outputs(paths, mock=True) == []


def test_verify_outputs_real_mode_reports_each_failed_gate(
    monkeypatch, tmp_path, make_corpus, small_split_settings
):
    paths = _finalized_corpus(monkeypatch, tmp_path, make_corpus, small_split_settings)
    monkeypatch.setattr(pipeline, "MAX_MEAN_DEVIATION", 0.1)
    monkeypatch.setattr(pipeline, "MAX_FEATURE_DEVIATION", 0.3)
    manifest = json.loads(paths.split_manifest.read_text())

    # The resplit above skipped the audit, which a publishable corpus must not have done.
    assert any("audit was skipped" in p for p in pipeline.verify_outputs(paths))

    # A last audit pass that still found leaks blocks the publish ...
    manifest["homology_audit"] = [{"pass": 0, "cross_partition_pairs": 3}]
    paths.split_manifest.write_text(json.dumps(manifest))
    assert any("3 cross-partition pairs remain" in p for p in pipeline.verify_outputs(paths))

    # ... a clean last pass, balanced features and met headline bars do not.
    manifest["homology_audit"] = [{"pass": 0, "cross_partition_pairs": 1}, {"pass": 1, "cross_partition_pairs": 0}]
    paths.split_manifest.write_text(json.dumps(manifest))
    assert pipeline.verify_outputs(paths) == []

    # Unbalanced partitions and an unmet headline bar are each reported.
    manifest["balance"]["features"]["size_gt_10"]["relative_deviation"]["discovery_val"] = 0.9  # reported, not gated
    paths.split_manifest.write_text(json.dumps(manifest))
    assert pipeline.verify_outputs(paths) == []
    off_feature = next(name for name in manifest["balance"]["features"] if not name.startswith("size_"))
    manifest["balance"]["features"][off_feature]["relative_deviation"]["discovery_val"] = 0.9
    paths.split_manifest.write_text(json.dumps(manifest))
    monkeypatch.setattr(pipeline, "CFG", dataclasses.replace(pipeline.CFG, min_sites_headline=10**9))
    problems = pipeline.verify_outputs(paths)
    assert any("discovery_val balance off target" in p for p in problems)
    assert any("phospho" in p and "headline bar" in p for p in problems)


def test_verify_outputs_catches_structural_damage(
    monkeypatch, tmp_path, make_corpus, small_split_settings
):
    paths = _finalized_corpus(monkeypatch, tmp_path, make_corpus, small_split_settings)
    corpus = pd.read_parquet(paths.corpus_parquet)
    corpus.loc[corpus["cluster_id"] == corpus["cluster_id"].iloc[0], "partition"] = None
    first_cluster = corpus.groupby("cluster_id").filter(lambda g: len(g) > 1)["cluster_id"].iloc[0]
    members = corpus.index[corpus["cluster_id"] == first_cluster]
    corpus.loc[members[0], "partition"] = "held_out"
    corpus.loc[members[1], "partition"] = "discovery_val"
    corpus.to_parquet(paths.corpus_parquet, index=False)

    problems = pipeline.verify_outputs(paths, mock=True)

    assert "proteins without a partition" in problems
    assert "a homology cluster spans several partitions" in problems


def test_assignment_stays_stable_across_audit_rounds(monkeypatch, make_corpus, small_split_settings):
    """Each audit round must repair the few leaking clusters, not reshuffle the whole partition:
    a from-scratch re-solve per round changes most proteins and never converges (measured on the
    real corpus: 549 -> 475 -> 174 leaking pairs)."""
    proteins, sites = make_corpus(n_clusters=200)
    rounds: list[pd.Series] = []

    def leaky_audit(frame, identity, tmp_dir):
        rounds.append(frame.set_index("uniprot_id")["partition"])
        if len(rounds) > 3:
            return []
        held = frame.loc[frame["partition"] == "held_out", "uniprot_id"].iloc[len(rounds)]
        train = frame.loc[frame["partition"] == "discovery_train", "uniprot_id"].iloc[len(rounds)]
        return [(train, held)]

    monkeypatch.setattr(clustering, "find_cross_partition_pairs", leaky_audit)

    settings = dataclasses.replace(small_split_settings, audit_passes=3)
    _, _, report = clustering.assign_partitions(proteins, sites, STRATUM_RESIDUES, settings)

    assert [e["cross_partition_pairs"] for e in report["homology_audit"]] == [1, 1, 1, 0]
    for before, after in itertools.pairwise(rounds):
        changed = (before != after.reindex(before.index)).mean()
        assert changed < 0.15, f"{changed:.0%} of proteins changed partition between audit rounds"


def test_audit_records_leaks_and_queries_per_boundary(monkeypatch, make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=150)

    def one_held_out_leak(frame, identity, tmp_dir):
        held = frame.loc[frame["partition"] == "held_out", "uniprot_id"].iloc[0]
        train = frame.loc[frame["partition"] == "discovery_train", "uniprot_id"].iloc[0]
        return [(train, held)]

    monkeypatch.setattr(clustering, "find_cross_partition_pairs", one_held_out_leak)

    _, _, report = clustering.assign_partitions(
        proteins, sites, STRATUM_RESIDUES, dataclasses.replace(small_split_settings, audit_passes=0)
    )

    entry = report["homology_audit"][0]
    assert entry["leaks"] == {"held_out": 1, "discovery_val": 0}
    assert entry["queries"]["held_out"] > 100 and entry["queries"]["discovery_val"] > 20


def test_merging_stops_early_once_every_partition_is_within_the_leak_target(
    monkeypatch, make_corpus, small_split_settings
):
    proteins, sites = make_corpus(n_clusters=200)
    calls = []

    def barely_leaky(frame, identity, tmp_dir):
        calls.append(1)
        held = frame.loc[frame["partition"] == "held_out", "uniprot_id"].iloc[0]
        train = frame.loc[frame["partition"] == "discovery_train", "uniprot_id"].iloc[0]
        return [(train, held)]  # 1 leak among ~400 held-out proteins: 0.25%

    monkeypatch.setattr(clustering, "find_cross_partition_pairs", barely_leaky)
    settings = dataclasses.replace(small_split_settings, audit_passes=5, leak_target=0.005)

    _, _, report = clustering.assign_partitions(proteins, sites, STRATUM_RESIDUES, settings)

    assert len(calls) == 1  # within target after the first audit: no merge rounds
    assert report["homology_audit"][0]["cross_partition_pairs"] == 1


def test_verify_outputs_gates_on_the_leak_fraction_per_partition(
    monkeypatch, tmp_path, make_corpus, small_split_settings
):
    paths = _finalized_corpus(monkeypatch, tmp_path, make_corpus, small_split_settings)
    monkeypatch.setattr(pipeline, "MAX_MEAN_DEVIATION", 0.1)
    monkeypatch.setattr(pipeline, "MAX_FEATURE_DEVIATION", 0.3)
    manifest = json.loads(paths.split_manifest.read_text())

    def audit(held_leaks, val_leaks):
        manifest["homology_audit"] = [{
            "pass": 3, "cross_partition_pairs": held_leaks + val_leaks,
            "queries": {"held_out": 1000, "discovery_val": 500},
            "leaks": {"held_out": held_leaks, "discovery_val": val_leaks},
        }]  # fmt: skip
        paths.split_manifest.write_text(json.dumps(manifest))
        return pipeline.verify_outputs(paths)

    assert audit(held_leaks=8, val_leaks=4) == []  # 0.8% and 0.8%: within the 1% tolerance
    problems = audit(held_leaks=8, val_leaks=9)  # val at 1.8%
    assert len(problems) == 1 and "discovery_val" in problems[0] and "1.80%" in problems[0]
    assert any("held_out" in p for p in audit(held_leaks=11, val_leaks=0))
