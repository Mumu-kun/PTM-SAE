"""Test suite for the partition-scoped streaming SafeTensors activation DataLoader."""

import json
from pathlib import Path

import safetensors.torch
import torch

from ptm_sae.training.dataset import (
    ActivationPartitionDataset,
    build_partition_dataloader,
    load_partition_ids,
)

HIDDEN_DIM = 4


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Builds a synthetic shard manifest (proteinA, proteinB in shard_0000; proteinC in
    shard_0001) plus a partition manifest assigning proteinA/proteinC to discovery_train,
    proteinB to discovery_val, and proteinD (never extracted) to held_out.

    Rows are filled with a per-protein constant marker value so tests can verify which
    protein's rows were actually streamed.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    corpus_dir = tmp_path / "processed"
    corpus_dir.mkdir()

    shard_0 = torch.cat(
        [
            torch.full((3, HIDDEN_DIM), 1.0),  # proteinA: discovery_train
            torch.full((2, HIDDEN_DIM), 2.0),  # proteinB: discovery_val
        ]
    )
    shard_1 = torch.full((4, HIDDEN_DIM), 3.0)  # proteinC: discovery_train

    safetensors.torch.save_file(
        {"activations": shard_0}, cache_dir / "shard_0000.safetensors"
    )
    safetensors.torch.save_file(
        {"activations": shard_1}, cache_dir / "shard_0001.safetensors"
    )

    manifest = {
        "version": "1.0",
        "total_tokens": 9,
        "shards": ["shard_0000.safetensors", "shard_0001.safetensors"],
        "entries": {
            "proteinA": {
                "uniprot_id": "proteinA",
                "length": 3,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 0,
                "end_offset": 3,
            },
            "proteinB": {
                "uniprot_id": "proteinB",
                "length": 2,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 3,
                "end_offset": 5,
            },
            "proteinC": {
                "uniprot_id": "proteinC",
                "length": 4,
                "shard_file": "shard_0001.safetensors",
                "start_offset": 0,
                "end_offset": 4,
            },
        },
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    partitions = [
        {"uniprot_id": "proteinA", "partition": "discovery_train"},
        {"uniprot_id": "proteinB", "partition": "discovery_val"},
        {"uniprot_id": "proteinC", "partition": "discovery_train"},
        {"uniprot_id": "proteinD", "partition": "held_out"},
    ]
    with open(corpus_dir / "proteins.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in partitions)

    return cache_dir, corpus_dir


def test_load_partition_ids_discovery_train(tmp_path):
    """Verify partition manifest filtering keeps only discovery_train UniProt IDs."""
    _, corpus_dir = _write_fixture(tmp_path)
    ids = load_partition_ids("discovery_train", corpus_dir=corpus_dir)
    assert ids == {"proteinA", "proteinC"}


def test_load_partition_ids_discovery_val(tmp_path):
    """Verify partition manifest filtering keeps only discovery_val UniProt IDs."""
    _, corpus_dir = _write_fixture(tmp_path)
    ids = load_partition_ids("discovery_val", corpus_dir=corpus_dir)
    assert ids == {"proteinB"}


def test_dataset_excludes_discovery_val_and_held_out(tmp_path):
    """Verify a discovery_train dataset streams strictly discovery_train rows, excluding
    discovery_val (proteinB, marker 2.0) entirely despite sharing a shard with discovery_train
    data."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    dataset = ActivationPartitionDataset(
        "discovery_train", cache_dir=cache_dir, corpus_dir=corpus_dir, shuffle=False
    )

    assert dataset.total_tokens == 7
    assert set(dataset.shard_files) == {"shard_0000.safetensors", "shard_0001.safetensors"}

    rows = list(dataset)
    assert len(rows) == 7

    markers = sorted(row[0].item() for row in rows)
    assert markers == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]


def test_dataset_discovery_val_partition(tmp_path):
    """Verify a discovery_val dataset streams strictly proteinB's rows."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    dataset = ActivationPartitionDataset(
        "discovery_val", cache_dir=cache_dir, corpus_dir=corpus_dir, shuffle=False
    )

    assert dataset.total_tokens == 2
    rows = list(dataset)
    markers = sorted(row[0].item() for row in rows)
    assert markers == [2.0, 2.0]


def test_dataset_shuffle_preserves_multiset(tmp_path):
    """Shuffling must reorder rows without dropping or duplicating any partition row."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    dataset = ActivationPartitionDataset(
        "discovery_train",
        cache_dir=cache_dir,
        corpus_dir=corpus_dir,
        shuffle=True,
        shuffle_buffer_size=2,
        seed=42,
    )

    rows = list(dataset)
    assert len(rows) == 7
    markers = sorted(row[0].item() for row in rows)
    assert markers == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]


def test_dataset_epoch_reseeds_shard_order(tmp_path):
    """Verify set_epoch changes the deterministic shard/row shuffle seed between epochs."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    dataset = ActivationPartitionDataset(
        "discovery_train",
        cache_dir=cache_dir,
        corpus_dir=corpus_dir,
        shuffle=True,
        shuffle_buffer_size=8,
        seed=0,
    )

    epoch_0 = [row.clone() for row in dataset]
    dataset.set_epoch(1)
    epoch_1 = [row.clone() for row in dataset]

    assert len(epoch_0) == len(epoch_1) == 7
    order_0 = [row[0].item() for row in epoch_0]
    order_1 = [row[0].item() for row in epoch_1]
    assert order_0 != order_1


def test_build_partition_dataloader_batches(tmp_path):
    """Verify the DataLoader convenience builder stacks rows into (batch, hidden_dim) tensors."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    dataset, loader = build_partition_dataloader(
        "discovery_train",
        batch_size=4,
        cache_dir=cache_dir,
        corpus_dir=corpus_dir,
        shuffle=False,
    )

    batches = list(loader)
    total_rows = sum(b.shape[0] for b in batches)
    assert total_rows == 7
    assert all(b.shape[1] == HIDDEN_DIM for b in batches)
    assert dataset.total_tokens == 7
