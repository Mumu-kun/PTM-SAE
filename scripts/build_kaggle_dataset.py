"""Stage discovery_train + discovery_val activations/corpus files from the HF Hub dataset into
a local directory, so it can be turned into a Kaggle Dataset — future Kaggle sessions then mount
it as a read-only input instead of re-downloading the whole corpus from HF every time.

Meant to be run INSIDE a Kaggle session (same repo-checkout + `pip install -e .` setup as
`notebooks/train_sae.ipynb`'s cells 1-3), writing into `/kaggle/working/`. Once it finishes,
use Kaggle's own "New Dataset" button on that kernel's Output tab — no `kaggle` CLI or
credentials needed for this path.

held_out is never fetched — this only pulls what Member 1's training loop actually reads. Note
that a shard file can hold a mix of proteins across all three partitions (extraction doesn't
shard per-partition), so a downloaded shard may still incidentally contain a held_out protein's
activations; that's harmless (nothing in the training loop ever looks up a held_out id) and
unavoidable without re-sharding the corpus.

`SafeTensorsReader`/`ActivationPartitionDataset` already skip any download attempt when the
target file exists locally (see `reader.py`), so no code changes are needed elsewhere: once the
resulting Kaggle Dataset is attached to a future session, point a training config's
`cache_dir`/`corpus_dir` at its mount path (`/kaggle/input/<dataset-slug>/activations`,
`.../corpus`) and set `remote_repo_id: null`.

Usage (inside a Kaggle notebook cell, after the repo/package setup cells):
    !python scripts/build_kaggle_dataset.py --output-dir /kaggle/working/discovery_corpus
"""

import argparse
import json
from pathlib import Path

from ptm_sae.extraction.hub import HfSyncClient
from ptm_sae.training.collapse_check import load_discovery_val_labels
from ptm_sae.training.dataset import load_partition_ids


def build_kaggle_dataset(
    output_dir: Path,
    repo_id: str,
    remote_subpath: str,
    remote_corpus_subpath: str,
) -> None:
    corpus_dir = output_dir / "corpus"
    activations_dir = output_dir / "activations"

    # Hydrates proteins.jsonl (partition IDs) and ptm_sites.jsonl (full file, filtered only at
    # read time) directly into corpus_dir as a side effect — no separate download logic needed.
    train_ids = load_partition_ids(
        "discovery_train", corpus_dir=corpus_dir, remote_repo_id=repo_id,
        remote_corpus_subpath=remote_corpus_subpath,
    )
    val_ids = load_partition_ids(
        "discovery_val", corpus_dir=corpus_dir, remote_repo_id=repo_id,
        remote_corpus_subpath=remote_corpus_subpath,
    )
    load_discovery_val_labels(
        corpus_dir=corpus_dir, remote_repo_id=repo_id, remote_corpus_subpath=remote_corpus_subpath,
    )
    needed_ids = train_ids | val_ids
    print(f"discovery_train + discovery_val: {len(needed_ids)} proteins")

    hub = HfSyncClient()
    manifest_path = activations_dir / "manifest.json"
    if not hub.fetch_manifest(repo_id, remote_subpath, manifest_path):
        raise FileNotFoundError(f"No manifest found at {repo_id}/{remote_subpath}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    needed_shards = sorted(
        {entry["shard_file"] for uid, entry in manifest["entries"].items() if uid in needed_ids}
    )
    print(f"{len(needed_shards)}/{len(manifest.get('shards', needed_shards))} shards needed")

    for i, shard_name in enumerate(needed_shards, 1):
        target = activations_dir / shard_name
        if target.exists():
            print(f"  [{i}/{len(needed_shards)}] {shard_name} already staged, skipping")
            continue
        print(f"  [{i}/{len(needed_shards)}] fetching {shard_name}...")
        hub.hydrate_shard(repo_id, remote_subpath, shard_name, target)

    print(
        f"\nStaged at {output_dir}. On Kaggle: use the kernel's Output tab -> \"New Dataset\" "
        f"to publish it, then point future configs' cache_dir/corpus_dir at the mounted "
        f"activations/ and corpus/ subfolders with remote_repo_id: null."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage discovery_train/discovery_val activations+corpus for a Kaggle Dataset"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Local staging directory")
    parser.add_argument("--repo-id", type=str, default="mustafa-muhaimin/ptm-sae-dataset")
    parser.add_argument(
        "--remote-subpath", type=str, default="activations/esm2_t33_650M_UR50D/layer_24"
    )
    parser.add_argument("--remote-corpus-subpath", type=str, default="corpus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_kaggle_dataset(output_dir, args.repo_id, args.remote_subpath, args.remote_corpus_subpath)


if __name__ == "__main__":
    main()
