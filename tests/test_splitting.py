"""Joint cluster-level partitioning: balance, determinism, floors, atomicity, failure modes."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ptm_sae.data.splitting import (
    PARTITIONS,
    SIZE_BINS,
    balance_report,
    build_cluster_features,
    partition_clusters,
)

STRATUM_RESIDUES = {"K": {"K"}, "ST": {"S", "T"}}


def test_build_cluster_features_columns_and_exclusions(make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=60)
    sites.loc[sites["type_pooled"] == "rare", "type_pooled"] = "K::other_PTM"

    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)

    assert set(features.names) >= {name for name, _, _ in SIZE_BINS}
    assert {"res_K", "res_ST", "sites_all", "proteins_with_site", "crosstalk_sites"} <= set(
        features.names
    )
    assert features.excluded_types == {"K::other_PTM": "pooled rare-type class"}
    assert {features.names[i] for i in features.type_rows} == {
        "type:phospho",
        "type:ubiq",
        "type:acetyl",
    }
    # Per-cluster tokens are the sum of member lengths, and the size bins partition them.
    assert features.tokens.sum() == proteins["length"].sum()
    size_rows = [features.names.index(name) for name, _, _ in SIZE_BINS]
    assert np.allclose(features.matrix[size_rows].sum(axis=0), features.tokens)


def test_lumpy_type_is_excluded(make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=60)
    one_cluster = sites["cluster_id"].iloc[0]
    sites.loc[sites["type_pooled"] == "ubiq", "cluster_id"] = one_cluster

    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)

    assert features.excluded_types["ubiq"] == "one cluster holds too large a share of its sites"


def test_partition_balances_every_feature_and_keeps_floors(make_corpus, small_split_settings):
    proteins, sites = make_corpus()
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)

    codes = partition_clusters(features, small_split_settings)
    report = balance_report(features, codes, small_split_settings)

    # Every cluster is assigned to exactly one partition.
    assert codes.shape == (len(features.cluster_ids),)
    assert set(np.unique(codes)) <= {0, 1, 2}
    # Hard token windows hold, and the targets are met closely on every balanced feature.
    tolerance = small_split_settings.token_tolerance
    for name, fraction in zip(PARTITIONS[1:], small_split_settings.fractions[1:], strict=True):
        assert abs(report["token_share"][name] - fraction) <= fraction * tolerance + 1e-9
        summary = report["summary"][name]
        # A 300-cluster corpus is coarse (val is ~3 mid-size clusters per bin); on the real corpus
        # the same code reaches <1%. The regression this guards is the old >100% skew.
        assert summary["mean_relative_deviation"] < 0.05, summary
        assert summary["max_relative_deviation"] < 0.15, summary
        assert summary["features_over_25pct"] == 0
    # Per-type floors are met in val and held_out.
    for ptm_type, parts in report["type_floors"].items():
        for name in PARTITIONS[1:]:
            achieved, floor = parts[name]
            assert achieved >= floor, (ptm_type, name)


def test_partition_is_deterministic(make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=150)
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)

    first = partition_clusters(features, small_split_settings)
    second = partition_clusters(features, small_split_settings)

    assert np.array_equal(first, second)


def test_val_has_the_same_cluster_size_mix_as_train(make_corpus, small_split_settings):
    """The defect this module exists to fix: val used to be all singletons."""
    proteins, sites = make_corpus()
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)

    codes = partition_clusters(features, small_split_settings)

    singleton = features.matrix[features.names.index("size_1")]
    share = {k: singleton[codes == k].sum() / features.tokens[codes == k].sum() for k in range(3)}
    assert abs(share[1] - share[0]) < 0.05
    assert abs(share[2] - share[0]) < 0.05


def test_infeasible_token_window_raises(small_split_settings):
    proteins = pd.DataFrame(
        {
            "uniprot_id": ["a", "b", "c"],
            "cluster_id": [0, 1, 2],
            "sequence": ["KKKK", "SSSS", "AAAA"],
            "length": [100, 100, 100],
        }
    )
    sites = pd.DataFrame(
        {"uniprot_id": ["a"], "cluster_id": [0], "type_pooled": ["phospho"], "crosstalk": [False]}
    )
    settings = dataclasses.replace(small_split_settings, min_type_sites=1)
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, settings)

    # Three indivisible 100-token clusters cannot hit a 10%/20% token target within +-3%.
    with pytest.raises(RuntimeError, match="no feasible assignment"):
        partition_clusters(features, settings)


def test_polish_from_an_already_polished_start_changes_nothing(make_corpus, small_split_settings):
    proteins, sites = make_corpus(n_clusters=150)
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)
    solved = partition_clusters(features, small_split_settings)

    again = partition_clusters(features, small_split_settings, start=solved)

    assert np.array_equal(again, solved)


def test_polish_steers_a_start_outside_the_token_windows_back_inside(make_corpus, small_split_settings):
    """After clusters are merged the previous assignment can violate the windows; polish repairs it
    instead of getting stuck (every single move used to look 'infeasible')."""
    proteins, sites = make_corpus(n_clusters=150)
    features = build_cluster_features(proteins, sites, STRATUM_RESIDUES, small_split_settings)
    everything_in_train = np.zeros(len(features.cluster_ids), dtype=int)

    codes = partition_clusters(features, small_split_settings, start=everything_in_train)

    report = balance_report(features, codes, small_split_settings)
    tolerance = small_split_settings.token_tolerance
    for name, fraction in zip(PARTITIONS[1:], small_split_settings.fractions[1:], strict=True):
        assert abs(report["token_share"][name] - fraction) <= fraction * tolerance + 1e-9
