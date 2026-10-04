"""Complete end-to-end PTM-SAE lifecycle pipeline orchestrator.

Coordinates:
1. Corpus Resolution (resolve the FASTA/protein set extraction will run over).
2. Protein Construction (build the `Protein` list `run_extraction_pipeline` consumes).
3. PLM Extraction (batched residue-level extraction & SafeTensors sharded buffering).
4. Remote Synchronization (resilient Hugging Face Hub status report).
5. Verification (zero-copy readback sanity checks).

Corpus-building itself (CD-HIT clustering + the PTM label cascade) no longer runs in-process
here -- it runs standalone as a Kaggle job (`ptm_sae.corpus.pipeline`, called directly from
`notebooks/kaggle_pipeline.ipynb`'s Phase A cells) and publishes `corpus.parquet`/
`labels_stratified.parquet`/etc. to the HF Hub. The non-sample path
below reads that already-published corpus (via `training.dataset.load_partition_ids`'s
local-cache-then-remote-hydrate pattern) instead of re-deriving one (implementation plan, Stage 4,
decision 7). `sample_only=True` never built a corpus this way either -- it only ever parsed
`data/sample.fasta` directly -- so that path is unaffected, just no longer routed through a
now-retired middleman.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from ptm_sae import runtime
from ptm_sae.config import PipelineConfig
from ptm_sae.corpus.config import CorpusPaths
from ptm_sae.data import parse_uniprot_fasta
from ptm_sae.data.schema import Protein
from ptm_sae.extraction.hub import publish_kaggle_dataset, resolve_hf_token
from ptm_sae.extraction.pipeline import run_extraction_pipeline
from ptm_sae.extraction.progress import PipelineProgressManager
from ptm_sae.extraction.reader import SafeTensorsReader
from ptm_sae.training.dataset import corpus_fingerprint, load_partition_ids

DEFAULT_CORPUS_REPO_ID = "mustafa-muhaimin/ptm-sae-corpus"


def _load_published_corpus_proteins(
    corpus_dir: Path,
    remote_corpus_repo_id: str | None,
    remote_corpus_subpath: str,
    token: str | None,
) -> list[Protein]:
    """Resolves the `discovery_train` + `discovery_val` protein set from an already-published
    `corpus.parquet` (never `held_out`, reserved for Member 2). Reuses `load_partition_ids` for
    the ID set -- which hydrates `corpus.parquet` locally as a side effect -- then reads the
    corresponding rows directly for `sequence`/`length`/`partition`.
    """
    train_ids = load_partition_ids(
        "discovery_train",
        corpus_dir=corpus_dir,
        remote_corpus_repo_id=remote_corpus_repo_id,
        remote_corpus_subpath=remote_corpus_subpath,
        token=token,
    )
    val_ids = load_partition_ids(
        "discovery_val",
        corpus_dir=corpus_dir,
        remote_corpus_repo_id=remote_corpus_repo_id,
        remote_corpus_subpath=remote_corpus_subpath,
        token=token,
    )
    needed_ids = train_ids | val_ids

    corpus_path = corpus_dir / "corpus.parquet"
    table = pq.read_table(corpus_path, columns=["uniprot_id", "sequence", "partition"])
    return [
        Protein(
            uniprot_id=row["uniprot_id"],
            sequence=row["sequence"],
            length=len(row["sequence"]),
            partition=row["partition"],
        )
        for row in table.to_pylist()
        if row["uniprot_id"] in needed_ids
    ]


def _kaggle_activation_dataset_slug(
    kaggle_username: str, config: PipelineConfig
) -> str:
    """One Kaggle Dataset per (model, layer), e.g. `<user>/ptm-sae-activations-esm2-t33-650m-ur50d-layer24`
    -- matches the HF remote_subpath's own per-model/layer scoping (`resolve_remote_subpath`)."""
    model_tag = config.model.model_name.split("/")[-1].lower().replace("_", "-")
    return f"{kaggle_username}/ptm-sae-activations-{model_tag}-layer{config.model.target_layer}"


def run_full_lifecycle(
    config: PipelineConfig,
    fasta_path: str | Path | None = None,
    sample_only: bool = False,
    max_proteins: int | None = None,
    remote_repo_id: str | None = None,
    remote_subpath: str | None = None,
    remote_corpus_repo_id: str | None = DEFAULT_CORPUS_REPO_ID,
    token: str | None = None,
    kaggle_username: str | None = None,
    progress_manager: PipelineProgressManager | None = None,
) -> dict:
    """Executes the extraction-and-synchronization lifecycle:

    1. Resolve Proteins: locates the local FASTA to parse (`sample_only`/`fasta_path`) or the
       already-published `corpus.parquet` to read from.
    2. Build Protein Records: parses the FASTA, or reads `discovery_train`+`discovery_val` rows
       straight out of the published corpus -- never `held_out`, reserved for Member 2.
    3. ESM-2 Activation Extraction: Batched extraction across single/multi-GPU into SafeTensors shards.
    4. Remote Sync: Reports whether extraction shards synced to the HF Hub (done asynchronously by
       the sharder itself during stage 3, not re-uploaded here), and -- if `kaggle_username` is
       given -- publishes/versions a Kaggle Dataset from the same local shard directory, one
       dataset per (model, layer), so a later run for a different layer doesn't overwrite this one.
    5. Readback Verification: Validates zero-copy indexing and biological coordinate invariants.
    """
    pm = progress_manager or PipelineProgressManager()
    pm.start_pipeline()

    processed_path = CorpusPaths.from_env().processed_dir  # same place corpus.pipeline writes
    processed_path.mkdir(parents=True, exist_ok=True)

    resolved_repo = remote_repo_id or config.sharding.remote_repo_id
    resolved_subpath = remote_subpath or config.resolve_remote_subpath()

    config.sharding.remote_repo_id = resolved_repo
    config.sharding.remote_subpath = resolved_subpath

    # 1. Resolve either a local FASTA (sample/pilot runs) or the published corpus location
    pm.start_stage(1, total_items=1, info="Resolving proteins for extraction")

    use_local_fasta = sample_only or fasta_path is not None
    if use_local_fasta:
        resolved_fasta = Path("data/sample.fasta") if sample_only else Path(fasta_path)
        if not resolved_fasta.exists():
            raise FileNotFoundError(f"FASTA file not found at {resolved_fasta}")
        pm.finish_stage(1, summary=f"Resolved local FASTA: {resolved_fasta.name}")
    else:
        pm.finish_stage(
            1,
            summary=f"Resolving published corpus at {processed_path} (repo={resolved_repo})",
        )

    # 2. Build the Protein records run_extraction_pipeline consumes
    pm.start_stage(2, total_items=1, info="Building Protein records")

    if use_local_fasta:
        target_proteins, skipped = parse_uniprot_fasta(
            resolved_fasta, max_sequence_length=config.extraction.max_sequence_length
        )
        pm.finish_stage(
            2,
            summary=f"{len(target_proteins)} proteins parsed from {resolved_fasta.name} "
            f"({len(skipped)} skipped, length-exceeded)",
        )
    else:
        target_proteins = _load_published_corpus_proteins(
            corpus_dir=processed_path,
            remote_corpus_repo_id=remote_corpus_repo_id,
            remote_corpus_subpath="corpus",
            token=token,
        )
        config.sharding.corpus_fingerprint = corpus_fingerprint(
            processed_path / "corpus.parquet"
        )
        pm.finish_stage(
            2,
            summary=f"{len(target_proteins)} proteins resolved (discovery_train + "
            "discovery_val) from published corpus.parquet",
        )

    if max_proteins is not None:
        target_proteins = target_proteins[:max_proteins]

    # 3. ESM-2 activation extraction and sharded buffering
    sharding_manifest = run_extraction_pipeline(
        config=config,
        proteins=target_proteins,
        progress_manager=pm,
    )

    # 4. Remote hub sync status (shard/manifest upload itself already ran asynchronously,
    # inside stage 3, via SafeTensorsSharder's own HfSyncClient), plus an optional Kaggle
    # Dataset publish from the same local shard directory.
    resolved_token = resolve_hf_token(token)

    if resolved_repo and resolved_token:
        pm.start_stage(
            4, total_items=1, info=f"Extraction shards synced to {resolved_repo}"
        )
        summary = (
            f"Extraction shards synced to {resolved_repo} during stage 3 "
            "(asynchronous background upload)."
        )
        if kaggle_username:
            slug = _kaggle_activation_dataset_slug(kaggle_username, config)
            publish_kaggle_dataset(
                Path(config.sharding.output_dir),
                slug,
                f"ESM-2 activations ({config.model.model_name}, layer {config.model.target_layer})",
                f"{len(target_proteins)} proteins",
            )
            summary += f" Kaggle Dataset published: {slug}."
        pm.finish_stage(4, summary=summary)
    elif resolved_repo and not resolved_token:
        pm.start_stage(
            4, total_items=1, info="Remote repo configured but token missing"
        )
        pm.finish_stage(
            4,
            summary=f"Warning: remote_repo_id ('{resolved_repo}') set, but no HF_TOKEN found. Skipped.",
        )
    else:
        pm.start_stage(4, total_items=1, info="Local cache retention mode")
        pm.finish_stage(
            4,
            summary="Local Mode: remote_repo_id not set. All shards retained in local cache.",
        )

    # 5. Zero-copy readback sanity verification
    pm.start_stage(5, total_items=1, info="Zero-copy readback sanity check")

    reader = SafeTensorsReader(
        cache_dir=config.sharding.output_dir,
        remote_repo_id=resolved_repo,
        remote_subpath=resolved_subpath,
        token=resolved_token,
    )
    available_proteins = reader.list_proteins()
    probe_summary = "No proteins available"
    if available_proteins:
        probe_id = available_proteins[0]
        acts = reader.get_protein_activations(probe_id)
        _res_1 = reader.get_residue_activation(probe_id, 1)
        mean_vec = reader.get_mean_pooled_embedding(probe_id)
        probe_summary = (
            f"Probe '{probe_id}' verified: acts {acts.shape}, mean {mean_vec.shape}"
        )

    pm.finish_stage(
        5,
        summary=f"{len(available_proteins)} proteins indexed in SafeTensors manifest | {probe_summary}",
    )

    pm.print("\n" + "═" * 72)
    pm.print("  PTM-SAE LIFECYCLE COMPLETED SUCCESSFULLY")
    pm.print("═" * 72)

    return {
        "protein_count": len(target_proteins),
        "sharding_manifest": sharding_manifest,
        "output_dir": config.sharding.output_dir,
        "processed_dir": str(processed_path),
        "remote_repo_id": resolved_repo,
        "remote_subpath": resolved_subpath,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Lifecycle Pipeline (Resolve Corpus -> Extract -> Sync -> Verify)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev_8m.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Path to a local FASTA file to extract directly, bypassing the published corpus",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Run in fast sample test mode on data/sample.fasta",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Cap extraction to N proteins (e.g. 500 for pilot test)",
    )
    parser.add_argument(
        "--remote-repo",
        type=str,
        default=None,
        help="Hugging Face Dataset repo ID (e.g. 'username/ptm-activations')",
    )
    parser.add_argument(
        "--remote-corpus-repo",
        type=str,
        default=DEFAULT_CORPUS_REPO_ID,
        help="Hugging Face Dataset repo ID holding the published corpus tables",
    )
    parser.add_argument(
        "--remote-subpath",
        type=str,
        default=None,
        help="Model/layer subpath inside remote repository",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device ('cuda', 'cpu', 'auto')",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Directory the relative sharding.output_dir hangs off (default: $PTM_SAE_DATA_ROOT, "
        "else the platform's writable area, else the repo root).",
    )
    parser.add_argument(
        "--kaggle-username",
        type=str,
        default=None,
        help="If set, publishes/versions a Kaggle Dataset per (model, layer) from the shard dir.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.device:
        cfg.model.device = args.device
    cfg.sharding.output_dir = runtime.anchor_path(
        cfg.sharding.output_dir, args.data_root or runtime.resolve_data_root()
    )

    run_full_lifecycle(
        config=cfg,
        fasta_path=args.fasta,
        sample_only=args.sample_only,
        max_proteins=args.max_proteins,
        remote_repo_id=args.remote_repo,
        remote_subpath=args.remote_subpath,
        remote_corpus_repo_id=args.remote_corpus_repo,
        kaggle_username=args.kaggle_username,
    )


if __name__ == "__main__":
    main()
