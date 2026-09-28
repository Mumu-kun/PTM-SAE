"""Build the PTM corpus end-to-end and publish it to both destinations in one run: Hugging Face
(the Remote Storage Authority `training/dataset.py`/`training/collapse_check.py` hydrate from)
and a Kaggle Dataset (for a future Kaggle session to mount directly, skipping the HF round-trip).

Meant to be run INSIDE a Kaggle session (same repo-checkout + `pip install -e .` setup as
`notebooks/kaggle_pipeline.ipynb`'s Phase A cells, which already call this script -- use that
notebook directly rather than assembling the setup by hand), with `cd-hit`/`cd-hit-2d` on PATH --
see `src/ptm_sae/corpus/README.md`. The full-scale build (network downloads across 8 sources +
CD-HIT + the label cascade over 18K+ proteins) runs exclusively here; local runs of this script
should pass `--mock` (small synthetic fixtures, no network/cd-hit-scale data needed) for
verification only.

Auth: HF_TOKEN (see `extraction.hub.resolve_hf_token`'s 5-tier cascade, e.g. Kaggle Secrets) and
KAGGLE_USERNAME/KAGGLE_KEY (the `kaggle` package's own env-var convention -- set both from Kaggle
Secrets the same way). KAGGLE_USERNAME/KAGGLE_KEY are only needed with `--kaggle-dataset-slug`.

Usage (inside a Kaggle notebook cell, after the repo/package setup cells):
    !python scripts/build_corpus.py --hf-repo-id mustafa-muhaimin/ptm-sae-dataset \
        --kaggle-dataset-slug <kaggle-username>/ptm-sae-corpus
"""

import argparse

from ptm_sae.corpus import pipeline
from ptm_sae.corpus.config import CorpusPaths
from ptm_sae.extraction.hub import HfSyncClient
from ptm_sae.extraction.hub import publish_kaggle_dataset as _publish_kaggle_dataset

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the PTM corpus and publish it to HF + (optionally) a Kaggle Dataset."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Small synthetic fixtures, for local verification.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore every cache.")
    parser.add_argument(
        "--hf-repo-id", type=str, default="mustafa-muhaimin/ptm-sae-dataset"
    )
    parser.add_argument("--hf-corpus-subpath", type=str, default="corpus")
    parser.add_argument(
        "--kaggle-dataset-slug",
        type=str,
        default=None,
        help="e.g. <kaggle-username>/ptm-sae-corpus. Omit to skip the Kaggle Dataset publish.",
    )
    parser.add_argument("--kaggle-version-message", type=str, default="Corpus rebuild")
    args = parser.parse_args()

    result = pipeline.run(mock=args.mock, force=args.force)
    print(
        f"Corpus build complete: {result['corpus']['partition'].value_counts().to_dict()}"
    )

    paths = CorpusPaths.from_env()

    print(f"\nUploading to HF ({args.hf_repo_id}/{args.hf_corpus_subpath}):")
    upload_to_hf(paths, args.hf_repo_id, args.hf_corpus_subpath)

    if args.kaggle_dataset_slug:
        print(f"\nPublishing Kaggle Dataset ({args.kaggle_dataset_slug}):")
        _publish_kaggle_dataset(
            paths.processed_dir,
            args.kaggle_dataset_slug,
            "PTM-SAE Corpus",
            args.kaggle_version_message,
        )
    else:
        print("\n--kaggle-dataset-slug not given, skipping Kaggle Dataset publish.")

    print(
        f"\nArtifacts staged at {paths.processed_dir} for either destination to re-read."
    )


if __name__ == "__main__":
    main()
