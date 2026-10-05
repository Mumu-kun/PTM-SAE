"""Streaming PyTorch IterableDataset over SafeTensors activation shards, scoped to a single
Corpus Partition (`discovery_train` or `discovery_val` — never `held_out`, which stays reserved
for Member 2's final evaluation)."""

import hashlib
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ptm_sae.extraction.hub import resolve_hf_token, retry_with_backoff
from ptm_sae.extraction.reader import SafeTensorsReader

Partition = Literal["discovery_train", "discovery_val"]


def corpus_fingerprint(corpus_parquet: str | Path) -> str:
    """SHA-256 identity of the corpus content that activations depend on: the (uniprot_id,
    sequence) of every discovery protein (train and val alike -- an activation does not depend on
    which side of discovery its protein sits on). `held_out` rows and label columns are left out,
    so rebuilding labels or re-splitting train/val does not invalidate extracted activations,
    while any change to the discovery set or a discovery sequence does."""
    table = pq.read_table(corpus_parquet, columns=["uniprot_id", "partition", "sequence"])
    rows = sorted(
        (r["uniprot_id"], r["sequence"])
        for r in table.to_pylist()
        if r["partition"] in ("discovery_train", "discovery_val")
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\t".join(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def partition_root(
    cache_dir: str | Path, remote_subpath: str | None, partition: str, partition_folders: bool
) -> tuple[Path, str | None]:
    """Local and remote root of a partition's activation shards: `<root>/<partition>` for the
    per-partition layout extraction writes, the shared root for the older mixed layout."""
    if not partition_folders:
        return Path(cache_dir), remote_subpath
    remote = f"{remote_subpath.rstrip('/')}/{partition}" if remote_subpath else partition
    return Path(cache_dir) / partition, remote


def hydrate_corpus_file(
    filename: str,
    corpus_dir: str | Path,
    remote_corpus_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
    required: bool = True,
) -> Path | None:
    """Ensures `filename` is cached locally under `corpus_dir`, hydrating it once from the
    Remote Storage Authority (`<remote_corpus_subpath>/<filename>`) if absent. Optional artifacts
    (`required=False`) that genuinely don't exist yet upstream (a real corpus build hasn't run
    yet) resolve to `None` instead of raising."""
    local_path = Path(corpus_dir) / filename

    if not local_path.exists() and remote_corpus_repo_id:
        resolved_token = resolve_hf_token(token)
        remote_path = f"{remote_corpus_subpath.rstrip('/')}/{filename}"

        def _download() -> str:
            return hf_hub_download(
                repo_id=remote_corpus_repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=resolved_token,
            )

        try:
            cached_file = retry_with_backoff(_download, max_retries=3, base_delay=2.0)
        except HfHubHTTPError:
            if required:
                raise
            return None
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, local_path)

    if not local_path.exists():
        if not required:
            return None
        raise FileNotFoundError(
            f"{filename} not found locally at {local_path} or on remote repository "
            f"{remote_corpus_repo_id}."
        )
    return local_path


def load_partition_ids(
    partition: Partition,
    corpus_dir: str | Path = "data/processed",
    remote_corpus_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> set[str]:
    """Resolves the set of UniProt IDs assigned to the given Corpus Partition, reading the local
    corpus.parquet (hydrated once from the Remote Storage Authority if absent)."""
    corpus_path = hydrate_corpus_file(
        "corpus.parquet", corpus_dir, remote_corpus_repo_id, remote_corpus_subpath, token
    )
    table = pq.read_table(corpus_path, columns=["uniprot_id", "partition"])
    return {
        record["uniprot_id"]
        for record in table.to_pylist()
        if record.get("partition") == partition
    }


class ActivationPartitionDataset(IterableDataset):
    """Streams ESM-2 residue activation vectors strictly from one Corpus Partition: single rows, or
    ready-made (batch_size, d) batches when `batch_size` is set (the training path; load it with
    `DataLoader(batch_size=None)`).

    Cross-references the activation shard manifest (proteins actually extracted) against the
    Corpus Partition manifest (`corpus.parquet`) to isolate exactly the requested partition,
    then hydrates shards on demand via SafeTensorsReader. Shard order is reshuffled every epoch
    and rows are shuffled in a bounded buffer for approximate i.i.d. sampling without
    materializing the full corpus in memory: the buffer fills to `shuffle_buffer_size` rows, is
    permuted once as a tensor and releases its first half, keeping the rest to mix with the next
    fill. Shards are striped evenly across DataLoader workers.
    """

    def __init__(
        self,
        partition: Partition,
        cache_dir: str | Path,
        remote_repo_id: str | None = None,
        remote_subpath: str | None = None,
        corpus_dir: str | Path = "data/processed",
        remote_corpus_repo_id: str | None = None,
        remote_corpus_subpath: str = "corpus",
        max_cached_shards: int | None = None,
        token: str | None = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = 65536,
        seed: int = 0,
        dtype: torch.dtype | None = torch.float32,
        partition_folders: bool = True,
        min_coverage: float = 0.0,
        batch_size: int | None = None,
    ):
        self.partition = partition
        self.batch_size = batch_size
        self.cache_dir, self.remote_subpath = partition_root(
            cache_dir, remote_subpath, partition, partition_folders
        )
        self.remote_repo_id = remote_repo_id
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
            remote_corpus_repo_id=remote_corpus_repo_id,
            remote_corpus_subpath=remote_corpus_subpath,
            token=token,
        )

        # Cross-reference shard manifest entries against the partition's IDs once, up front.
        probe_reader = SafeTensorsReader(
            cache_dir=self.cache_dir,
            remote_repo_id=remote_repo_id,
            remote_subpath=self.remote_subpath,
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

        # Stale-activation guards: the manifest's corpus stamp must match the corpus on disk, and
        # partial coverage is never silent (a run on 74% of discovery_train looks fine otherwise).
        manifest_fingerprint = probe_reader.manifest.get("corpus_fingerprint")
        if manifest_fingerprint:
            current = corpus_fingerprint(Path(corpus_dir) / "corpus.parquet")
            if current != manifest_fingerprint:
                raise ValueError(
                    f"Activations in {self.cache_dir} were extracted from a different corpus "
                    f"({manifest_fingerprint[:12]}...) than {Path(corpus_dir) / 'corpus.parquet'} "
                    f"({current[:12]}...). Re-extract against the current corpus."
                )
        covered = sum(len(ids) for ids in shard_entries.values())
        if covered < min_coverage * len(partition_ids):
            raise ValueError(
                f"Only {covered}/{len(partition_ids)} {partition} proteins ({covered / len(partition_ids):.1%}) have "
                f"activations in {self.cache_dir}, below min_partition_coverage={min_coverage:.0%}: a run on a "
                "fraction of the partition looks fine otherwise. Extract the rest, or lower the threshold on purpose "
                "(the 8M pilot configs do)."
            )
        if covered < len(partition_ids):
            warnings.warn(
                f"Only {covered}/{len(partition_ids)} {partition} proteins "
                f"({covered / len(partition_ids):.1%}) have activations in {self.cache_dir}"
                + ("" if manifest_fingerprint else " (manifest predates corpus stamping)")
                + ".",
                stacklevel=2,
            )

        if not shard_entries:
            raise ValueError(
                f"No {partition} activation entries found — check that the shard manifest "
                "and corpus.parquet partition manifest reference the same UniProt IDs."
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

        # Shuffle in tensor blocks: one randperm per buffer fill instead of a Python draw per row.
        # Unshuffled (validation), blocks are released whole and in order.
        held: list[torch.Tensor] = []
        held_rows = 0
        pending: list[torch.Tensor] = []  # released rows not yet cut into a batch
        pending_rows = 0

        def release(final: bool) -> torch.Tensor:
            nonlocal held, held_rows
            block = torch.cat(held)
            if self.shuffle:
                block = block[torch.randperm(len(block), generator=row_rng)]
            released = len(block) if final or not self.shuffle else len(block) // 2
            held, held_rows = [block[released:]], len(block) - released
            return block[:released]

        def cast(rows: torch.Tensor) -> torch.Tensor:  # after the shuffle: the shards are fp16, so it moves half the bytes
            return rows if self.dtype is None else rows.to(self.dtype)

        def batches(block: torch.Tensor) -> Iterator[torch.Tensor]:
            nonlocal pending, pending_rows
            if self.batch_size is None:
                yield from cast(block).unbind(dim=0)
                return
            pending.append(block)
            pending_rows += len(block)
            if pending_rows < self.batch_size:
                return
            stacked = torch.cat(pending)
            full = len(stacked) // self.batch_size * self.batch_size
            for batch in stacked[:full].split(self.batch_size):
                yield cast(batch)
            pending, pending_rows = [stacked[full:]], len(stacked) - full

        try:
            for shard_file in worker_shards:
                for uniprot_id in self.shard_entries[shard_file]:
                    protein_acts = reader.get_protein_activations(uniprot_id)
                    held.append(protein_acts)
                    held_rows += len(protein_acts)
                    if held_rows >= self.shuffle_buffer_size:
                        yield from batches(release(final=False))

            if held_rows:
                yield from batches(release(final=True))
            if pending_rows:  # the last, shorter batch (also the whole stream when it is under one batch)
                yield cast(torch.cat(pending))
        finally:
            reader.close()


def build_partition_dataloader(
    partition: Partition,
    batch_size: int,
    cache_dir: str | Path,
    remote_repo_id: str | None = None,
    remote_subpath: str | None = None,
    corpus_dir: str | Path = "data/processed",
    remote_corpus_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    num_workers: int = 0,
    max_cached_shards: int | None = None,
    token: str | None = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = 65536,
    seed: int = 0,
    dtype: torch.dtype | None = torch.float32,
    partition_folders: bool = True,
    min_coverage: float = 0.0,
) -> tuple[ActivationPartitionDataset, DataLoader]:
    """Convenience builder wiring an ActivationPartitionDataset into a torch DataLoader.

    Returns both the dataset (call `.set_epoch(n)` on it before each epoch to reseed shuffling)
    and the DataLoader itself, which yields (batch_size, hidden_dim) batches (the last one may be shorter).
    """
    dataset = ActivationPartitionDataset(
        partition=partition,
        cache_dir=cache_dir,
        remote_repo_id=remote_repo_id,
        remote_subpath=remote_subpath,
        corpus_dir=corpus_dir,
        remote_corpus_repo_id=remote_corpus_repo_id,
        remote_corpus_subpath=remote_corpus_subpath,
        max_cached_shards=max_cached_shards,
        token=token,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        dtype=dtype,
        partition_folders=partition_folders,
        min_coverage=min_coverage,
        batch_size=batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,  # the dataset already yields (batch_size, d) tensors
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader
