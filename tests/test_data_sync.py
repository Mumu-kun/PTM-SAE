"""Offline tests for the explicit data sync (missing-only fetch, verify, force, partitions)."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ptm_sae.data.sync import sync_data
from ptm_sae.extraction.hub import HfSyncClient
from ptm_sae.training.config import SAETrainingConfig

REPO = "mock/activations"
CORPUS_REPO = "mock/corpus"
SUBPATH = "activations/layer_4"


class FakeHub:
    """Stands in for HfSyncClient with one directory per 'remote' dataset repo."""

    build_repo_path = staticmethod(HfSyncClient.build_repo_path)

    def __init__(self, remote_root: Path):
        self.remote_root = remote_root  # remote_root/<repo name>/<path in repo>
        self.downloads: list[tuple[str, str]] = []

    def get_remote_files_info(self, repo_id, remote_paths):
        info = {}
        for path in remote_paths:
            f = self.remote_root / repo_id / path
            if f.exists():
                info[path] = (f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest())
        return info

    def hydrate_shard(self, repo_id, subpath, shard_name, target_path, expected_bytes=None):
        source = self.remote_root / repo_id / self.build_repo_path(subpath, shard_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        self.downloads.append((repo_id, shard_name))
        return target_path


@pytest.fixture
def remote_and_config(tmp_path):
    """A remote with 2 shards (train in shard_0000, val in shard_0001) + corpus tables."""
    remote = tmp_path / "remote"
    activations_repo, corpus_repo = remote / REPO, remote / CORPUS_REPO
    (activations_repo / SUBPATH / "shards").mkdir(parents=True)
    (corpus_repo / "corpus").mkdir(parents=True)

    for i in (0, 1):
        (activations_repo / SUBPATH / "shards" / f"shard_000{i}.safetensors").write_bytes(b"x" * (100 + i))
    entries = {
        "protA": {"uniprot_id": "protA", "length": 5, "shard_file": "shard_0000.safetensors"},
        "protC": {"uniprot_id": "protC", "length": 5, "shard_file": "shard_0001.safetensors"},
    }
    (activations_repo / SUBPATH / "manifest.json").write_text(
        json.dumps({"shards": list(), "entries": entries}), encoding="utf-8"
    )
    rows = [
        {"uniprot_id": "protA", "partition": "discovery_train", "sequence": "A"},
        {"uniprot_id": "protC", "partition": "discovery_val", "sequence": "A"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), corpus_repo / "corpus" / "corpus.parquet")
    pq.write_table(pa.Table.from_pylist(rows), corpus_repo / "corpus" / "labels_stratified.parquet")

    config = SAETrainingConfig(
        total_steps=10,
        remote_repo_id=REPO,
        remote_corpus_repo_id=CORPUS_REPO,
        remote_subpath=SUBPATH,
        cache_dir="cache",
        corpus_dir="processed",
        partition_folders=False,
    ).with_data_root(tmp_path / "local")
    return FakeHub(remote), config


def test_per_partition_layout_syncs_each_partitions_own_folder(remote_and_config):
    hub, config = remote_and_config
    activations = hub.remote_root / REPO / SUBPATH
    for partition, (shard, protein) in {"discovery_train": ("shard_0000", "protA"), "discovery_val": ("shard_0001", "protC")}.items():
        (activations / partition / "shards").mkdir(parents=True)
        (activations / partition / "shards" / f"{shard}.safetensors").write_bytes(b"x" * 100)
        entry = {protein: {"uniprot_id": protein, "length": 5, "shard_file": f"{shard}.safetensors"}}
        (activations / partition / "manifest.json").write_text(json.dumps({"shards": [], "entries": entry}), encoding="utf-8")

    report = sync_data(config.model_copy(update={"partition_folders": True}), hub=hub)

    assert sorted(report.fetched) == [
        "corpus.parquet",
        "discovery_train/manifest.json",
        "discovery_train/shard_0000.safetensors",
        "discovery_val/manifest.json",
        "discovery_val/shard_0001.safetensors",
        "labels_stratified.parquet",
    ]
    assert (Path(config.cache_dir) / "discovery_val" / "shard_0001.safetensors").exists()


def test_default_fetches_only_what_is_missing(remote_and_config):
    hub, config = remote_and_config

    first = sync_data(config, hub=hub)
    assert sorted(first.fetched) == [
        "corpus.parquet",
        "labels_stratified.parquet",
        "manifest.json",
        "shard_0000.safetensors",
        "shard_0001.safetensors",
    ]
    assert first.absent_remote == ["exclusion_mask.parquet", "gold_negatives_nglyco.parquet"]

    hub.downloads.clear()
    second = sync_data(config, hub=hub)
    assert second.fetched == {}
    assert hub.downloads == []


def test_partition_filter_limits_shards(remote_and_config):
    hub, config = remote_and_config

    report = sync_data(config, partitions=("discovery_train",), hub=hub)

    assert "shard_0000.safetensors" in report.fetched
    assert "shard_0001.safetensors" not in report.fetched


def test_what_corpus_skips_activations(remote_and_config):
    hub, config = remote_and_config

    report = sync_data(config, what="corpus", hub=hub)

    assert sorted(report.fetched) == ["corpus.parquet", "labels_stratified.parquet"]


def test_truncated_shard_is_refetched_even_without_verify(remote_and_config):
    hub, config = remote_and_config
    sync_data(config, hub=hub)
    shard = Path(config.cache_dir) / "shard_0000.safetensors"
    shard.write_bytes(b"x" * 10)

    report = sync_data(config, hub=hub)

    assert report.fetched == {"shard_0000.safetensors": "size mismatch"}
    assert shard.stat().st_size == 100


def test_verify_catches_same_size_corruption(remote_and_config):
    hub, config = remote_and_config
    sync_data(config, hub=hub)
    shard = Path(config.cache_dir) / "shard_0000.safetensors"
    shard.write_bytes(b"y" * 100)  # same size, wrong content

    assert sync_data(config, hub=hub).fetched == {}  # size check alone cannot see it
    report = sync_data(config, verify=True, hub=hub)

    assert report.fetched == {"shard_0000.safetensors": "checksum mismatch"}
    assert shard.read_bytes() == b"x" * 100


def test_force_refetches_everything_selected(remote_and_config):
    hub, config = remote_and_config
    sync_data(config, hub=hub)

    report = sync_data(config, what="corpus", force=True, hub=hub)

    assert set(report.fetched.values()) == {"forced"}
    assert sorted(report.fetched) == ["corpus.parquet", "labels_stratified.parquet"]


def test_missing_required_remote_file_raises(remote_and_config):
    hub, config = remote_and_config
    (hub.remote_root / CORPUS_REPO / "corpus" / "labels_stratified.parquet").unlink()

    with pytest.raises(FileNotFoundError, match=r"labels_stratified\.parquet"):
        sync_data(config, hub=hub)


def test_activations_and_corpus_come_from_their_own_repos(remote_and_config):
    hub, config = remote_and_config

    sync_data(config, hub=hub)

    repos_by_file = dict((name, repo) for repo, name in hub.downloads)
    assert repos_by_file["corpus.parquet"] == CORPUS_REPO
    assert repos_by_file["manifest.json"] == REPO
    assert repos_by_file["shard_0000.safetensors"] == REPO


def test_requires_remote_repos(remote_and_config):
    hub, config = remote_and_config

    with pytest.raises(ValueError, match="remote_corpus_repo_id"):
        sync_data(config.model_copy(update={"remote_corpus_repo_id": None}), hub=hub)

    no_activations = config.model_copy(update={"remote_repo_id": None})
    with pytest.raises(ValueError, match="remote_repo_id"):
        sync_data(no_activations, hub=hub)
    assert sorted(sync_data(no_activations, what="corpus", hub=hub).fetched) == [
        "corpus.parquet",
        "labels_stratified.parquet",
    ]
