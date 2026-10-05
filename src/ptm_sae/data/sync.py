"""Explicit, resumable download and verification of the corpus tables and activation shards a
training run reads, so a run never has to hydrate them over a flaky link mid-training.

Usage:
    uv run python -m ptm_sae.data.sync --config configs/train_topk_baseline.yaml
    uv run python -m ptm_sae.data.sync --config ... --verify          # re-hash, repair bad files
    uv run python -m ptm_sae.data.sync --config ... --force --what corpus
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ptm_sae import runtime
from ptm_sae.extraction.hub import HfSyncClient, compute_sha256
from ptm_sae.training.config import SAETrainingConfig
from ptm_sae.training.dataset import (
    ActivationPartitionDataset,
    Partition,
    partition_root,
)

# (filename, required) — mirrors what the training loop and its canaries read from corpus_dir.
CORPUS_FILES = (
    ("corpus.parquet", True),
    ("labels_stratified.parquet", True),
    ("exclusion_mask.parquet", False),
    ("gold_negatives_nglyco.parquet", False),
)

What = Literal["corpus", "activations", "all"]


@dataclass
class SyncTarget:
    """One file to keep in sync: where it lives remotely, where it goes locally."""

    remote_subpath: str | None
    name: str
    local_path: Path
    required: bool = True
    label: str = ""  # report key when `name` alone is ambiguous (the same file name in two partitions)


@dataclass
class SyncReport:
    fetched: dict[str, str] = field(default_factory=dict)  # name -> why it was (re)fetched
    up_to_date: list[str] = field(default_factory=list)
    absent_remote: list[str] = field(default_factory=list)  # optional files not on the Hub


def _sync_targets(
    hub: HfSyncClient,
    repo_id: str | None,
    targets: list[SyncTarget],
    report: SyncReport,
    verify: bool,
    force: bool,
) -> None:
    remote_paths = {t.name: hub.build_repo_path(t.remote_subpath, t.name) for t in targets}
    remote_info = hub.get_remote_files_info(repo_id, list(remote_paths.values()))

    for target in targets:
        info = remote_info.get(remote_paths[target.name])
        if info is None and not target.required:
            report.absent_remote.append(target.name)
            continue
        if info is None:
            raise FileNotFoundError(
                f"{remote_paths[target.name]} not found on remote repository {repo_id}."
            )

        size, sha256 = info
        key = target.label or target.name
        if force:
            reason = "forced"
        elif not target.local_path.exists():
            reason = "missing"
        elif target.local_path.stat().st_size != size:
            reason = "size mismatch"
        elif verify and sha256 and compute_sha256(target.local_path) != sha256:
            reason = "checksum mismatch"
        else:
            report.up_to_date.append(key)
            continue

        # hydrate_shard writes atomically and replaces the old file only once the new one passed
        # its size check, so a failed re-fetch never leaves a half-deleted local copy behind.
        hub.hydrate_shard(
            repo_id,
            target.remote_subpath,
            target.name,
            target.local_path,
            expected_bytes=size,
        )
        report.fetched[key] = reason


def sync_data(
    config: SAETrainingConfig,
    what: What = "all",
    partitions: tuple[Partition, ...] = ("discovery_train", "discovery_val"),
    verify: bool = False,
    force: bool = False,
    hub: HfSyncClient | None = None,
) -> SyncReport:
    """Brings the local corpus tables and/or the activation shards of `partitions` up to date.

    Default fetches only what is missing or has the wrong size. `verify` additionally re-hashes
    every local file against the Hub's SHA-256 and re-fetches mismatches; `force` re-fetches
    everything selected regardless. `config` should already be anchored via `with_data_root`.
    """
    if not config.remote_corpus_repo_id:
        raise ValueError("config.remote_corpus_repo_id must be set to sync the corpus from the Hub.")
    if what != "corpus" and not config.remote_repo_id:
        raise ValueError("config.remote_repo_id must be set to sync activations from the Hub.")
    hub = hub or HfSyncClient()
    report = SyncReport()

    def sync(repo_id: str | None, targets: list[SyncTarget]) -> None:
        _sync_targets(hub, repo_id, targets, report, verify, force)

    # 1. Corpus tables. corpus.parquet is always needed to resolve partition shards below.
    corpus_dir = Path(config.corpus_dir)
    corpus_files = CORPUS_FILES if what in ("corpus", "all") else CORPUS_FILES[:1]
    sync(
        config.remote_corpus_repo_id,
        [
            SyncTarget(config.remote_corpus_subpath, name, corpus_dir / name, required)
            for name, required in corpus_files
        ]
    )
    if what == "corpus":
        return report

    # 2. Per partition: its shard manifest, then only the shards that hold its proteins.
    cache_dir = Path(config.cache_dir)
    sub = config.remote_subpath.rstrip("/") if config.remote_subpath else None
    for partition in partitions:
        local_root, remote_root = partition_root(cache_dir, sub, partition, config.partition_folders)
        prefix = f"{partition}/" if config.partition_folders else ""
        sync(
            config.remote_repo_id,
            [SyncTarget(remote_root, "manifest.json", local_root / "manifest.json", label=f"{prefix}manifest.json")],
        )

        # Everything is local now, so no remote_repo_id: this only cross-references the
        # manifest against the partition's IDs.
        dataset = ActivationPartitionDataset(
            partition,
            cache_dir=cache_dir,
            corpus_dir=corpus_dir,
            remote_repo_id=None,
            partition_folders=config.partition_folders,
        )
        shards_subpath = f"{remote_root}/shards" if remote_root else "shards"
        sync(
            config.remote_repo_id,
            [
                SyncTarget(shards_subpath, name, local_root / name, label=f"{prefix}{name}")
                for name in sorted(dataset.shard_files)
            ],
        )
    return report


def main():
    runtime.ensure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Download/verify the corpus tables and activation shards a training config reads."
    )
    parser.add_argument("--config", type=str, required=True, help="Training YAML (repo, subpaths, dirs)")
    parser.add_argument("--what", choices=["corpus", "activations", "all"], default="all")
    parser.add_argument(
        "--partition",
        dest="partitions",
        action="append",
        choices=["discovery_train", "discovery_val"],
        help="Partition(s) whose shards to sync, repeatable (default: both).",
    )
    parser.add_argument("--verify", action="store_true", help="Re-hash local files; re-fetch mismatches.")
    parser.add_argument("--force", action="store_true", help="Re-fetch everything selected.")
    parser.add_argument("--data-root", type=str, default=None, help="See ptm_sae.training.train --data-root.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = (
        SAETrainingConfig.from_yaml(args.config)
        .with_overrides(args.overrides)
        .with_data_root(args.data_root)
    )
    report = sync_data(
        config,
        what=args.what,
        partitions=tuple(args.partitions or ("discovery_train", "discovery_val")),
        verify=args.verify,
        force=args.force,
    )

    for name, reason in report.fetched.items():
        print(f"[fetched: {reason}] {name}")
    for name in report.absent_remote:
        print(f"[not on Hub, optional] {name}")
    print(f"[sync] {len(report.fetched)} fetched, {len(report.up_to_date)} already up to date.")


if __name__ == "__main__":
    main()
