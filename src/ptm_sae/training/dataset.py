"""Streaming PyTorch IterableDataset over SafeTensors activation shards, scoped to a single
Corpus Partition (`discovery_train` or `discovery_val` — never `held_out`, which stays reserved
for Member 2's final evaluation)."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ptm_sae.extraction.hub import resolve_hf_token, retry_with_backoff
from ptm_sae.extraction.reader import SafeTensorsReader

Partition = Literal["discovery_train", "discovery_val"]


def load_partition_ids(
    partition: Partition,
    corpus_dir: str | Path = "data/processed",
    remote_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> set[str]:
    """Resolves the set of UniProt IDs assigned to the given Corpus Partition.

    Reads the local proteins.jsonl if cached, otherwise hydrates it once from the
    Remote Storage Authority (`<remote_corpus_subpath>/proteins.jsonl`).
    """
    local_path = Path(corpus_dir) / "proteins.jsonl"

    if not local_path.exists() and remote_repo_id:
        resolved_token = resolve_hf_token(token)
        remote_path = f"{remote_corpus_subpath.rstrip('/')}/proteins.jsonl"

        def _download() -> str:
            return hf_hub_download(
                repo_id=remote_repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=resolved_token,
            )

        cached_file = retry_with_backoff(_download, max_retries=3, base_delay=2.0)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"proteins.jsonl not found locally at {local_path} or on remote repository {remote_repo_id}."
        )

    partition_ids: set[str] = set()
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("partition") == partition:
                partition_ids.add(record["uniprot_id"])

    return partition_ids


class ActivationPartitionDataset(IterableDataset):
    """Streams individual ESM-2 residue activation vectors strictly from one Corpus Partition.

    Cross-references the activation shard manifest (proteins actually extracted) against the
    Corpus Partition manifest (`proteins.jsonl`) to isolate exactly the requested partition,
    then hydrates shards on demand via SafeTensorsReader. Shard order is reshuffled every epoch
    and rows are drawn from a bounded shuffle buffer for approximate i.i.d. sampling without
    materializing the full corpus in memory. Shards are striped evenly across DataLoader workers.
    """

    def __init__(
        self,
        partition: Partition,
        cache_dir: str | Path,
        remote_repo_id: str | None = None,
        remote_subpath: str | None = None,
        corpus_dir: str | Path = "data/processed",
        remote_corpus_subpath: str = "corpus",
        max_cached_shards: int | None = None,
        token: str | None = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = 65536,
        seed: int = 0,
        dtype: torch.dtype | None = torch.float32,
    ):
        self.partition = partition
        self.cache_dir = cache_dir
        self.remote_repo_id = remote_repo_id
        self.remote_subpath = remote_subpath
        self.max_cached_shards = max_cached_shards
        self.token = token
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.dtype = dtype
        self.epoch = 0

        partition_ids = load_partition_ids(
            partition=partition,
            corpus_dir=corpus_dir,
            remote_repo_id=remote_repo_id,
            remote_corpus_subpath=remote_corpus_subpath,
            token=token,
        )

        # Cross-reference shard manifest entries against the partition's IDs once, up front.
        probe_reader = SafeTensorsReader(
            cache_dir=cache_dir,
            remote_repo_id=remote_repo_id,
            remote_subpath=remote_subpath,
            max_cached_shards=max_cached_shards,
            token=token,
        )
        shard_entries: dict[str, list[str]] = {}
        total_tokens = 0
        for uniprot_id, entry in probe_reader.entries.items():
            if uniprot_id not in partition_ids:
                continue
            shard_entries.setdefault(entry["shard_file"], []).append(uniprot_id)
            total_tokens += entry["length"]
        probe_reader.close()

        if not shard_entries:
            raise ValueError(
                f"No {partition} activation entries found — check that the shard manifest "
                "and proteins.jsonl partition manifest reference the same UniProt IDs."
            )

        self.shard_entries = shard_entries
        self.shard_files = sorted(shard_entries.keys())
        self.total_tokens = total_tokens

    def set_epoch(self, epoch: int) -> None:
        """Reseeds shard and row shuffling for a new training epoch. Call before each epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_tokens

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        shard_order = self.shard_files
        if self.shuffle:
            shard_rng = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(shard_order), generator=shard_rng).tolist()
            shard_order = [shard_order[i] for i in perm]

        # Stripe shards across workers so no shard is read by more than one worker.
        worker_shards = shard_order[worker_id::num_workers]

        reader = SafeTensorsReader(
            cache_dir=self.cache_dir,
            remote_repo_id=self.remote_repo_id,
            remote_subpath=self.remote_subpath,
            max_cached_shards=self.max_cached_shards,
            token=self.token,
        )
        row_rng = torch.Generator().manual_seed(self.seed + self.epoch + worker_id + 1)
        buffer: list[torch.Tensor] = []

        try:
            for shard_file in worker_shards:
                for uniprot_id in self.shard_entries[shard_file]:
                    protein_acts = reader.get_protein_activations(uniprot_id)
                    if self.dtype is not None:
                        protein_acts = protein_acts.to(self.dtype)

                    if not self.shuffle:
                        yield from protein_acts.unbind(dim=0)
                        continue

                    for row in protein_acts.unbind(dim=0):
                        buffer.append(row)
                        if len(buffer) >= self.shuffle_buffer_size:
                            idx = int(torch.randint(len(buffer), (1,), generator=row_rng))
                            buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                            yield buffer.pop()

            if self.shuffle:
                while buffer:
                    idx = int(torch.randint(len(buffer), (1,), generator=row_rng))
                    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                    yield buffer.pop()
        finally:
            reader.close()


def build_partition_dataloader(
    partition: Partition,
    batch_size: int,
    cache_dir: str | Path,
    remote_repo_id: str | None = None,
    remote_subpath: str | None = None,
    corpus_dir: str | Path = "data/processed",
    remote_corpus_subpath: str = "corpus",
    num_workers: int = 0,
    max_cached_shards: int | None = None,
    token: str | None = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = 65536,
    seed: int = 0,
    dtype: torch.dtype | None = torch.float32,
) -> tuple[ActivationPartitionDataset, DataLoader]:
    """Convenience builder wiring an ActivationPartitionDataset into a torch DataLoader.

    Returns both the dataset (call `.set_epoch(n)` on it before each epoch to reseed shuffling)
    and the DataLoader itself, which stacks individual residue rows into (batch_size, hidden_dim).
    """
    dataset = ActivationPartitionDataset(
        partition=partition,
        cache_dir=cache_dir,
        remote_repo_id=remote_repo_id,
        remote_subpath=remote_subpath,
        corpus_dir=corpus_dir,
        remote_corpus_subpath=remote_corpus_subpath,
        max_cached_shards=max_cached_shards,
        token=token,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        dtype=dtype,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader
