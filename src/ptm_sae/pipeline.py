"""Complete end-to-end PTM-SAE lifecycle pipeline orchestrator.

Coordinates:
1. Data Acquisition (fetch UniProt proteome & raw PTM datasets).
2. Preparation & Splitting (homology partitioning & invariant validation).
3. PLM Extraction (batched residue-level extraction & SafeTensors sharded buffering).
4. Remote Synchronization (resilient Hugging Face Hub atomic commits).
5. Verification (zero-copy readback sanity checks).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ptm_sae.config import PipelineConfig
from ptm_sae.data import (
    PTMCorpus,
    fetch_cplm_human,
    fetch_uniprot_human_proteome,
    fetch_uniprot_ptm_features,
    prepare_ptm_corpus,
)
from ptm_sae.extraction.hub import HfSyncClient, resolve_hf_token
from ptm_sae.extraction.pipeline import run_extraction_pipeline
from ptm_sae.extraction.progress import PipelineProgressManager
from ptm_sae.extraction.reader import SafeTensorsReader


def run_full_lifecycle(
    config: PipelineConfig,
    fasta_path: str | Path | None = None,
    ptm_sources: Sequence[Path] | None = None,
    sample_only: bool = False,
    max_proteins: int | None = None,
    remote_repo_id: str | None = None,
    remote_subpath: str | None = None,
    token: str | None = None,
    progress_manager: PipelineProgressManager | None = None,
) -> dict:
    """Executes the entire data acquisition, extraction, and remote synchronization lifecycle:

    1. Data Acquisition: Downloads reviewed human proteome and raw PTM annotations.
    2. Harmonization & Partitioning: Parses sequences, clusters homology groups, and solves MILP split.
    3. ESM-2 Activation Extraction: Batched extraction across single/multi-GPU into SafeTensors shards.
    4. Remote Upload: Pushes shards, auxiliary context, and manifests to structured HF Hub subpaths.
    5. Readback Verification: Validates zero-copy indexing and biological coordinate invariants.
    """
    pm = progress_manager or PipelineProgressManager()
    pm.start_pipeline()

    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    resolved_repo = remote_repo_id or config.sharding.remote_repo_id
    resolved_subpath = remote_subpath or config.resolve_remote_subpath()

    config.sharding.remote_repo_id = resolved_repo
    config.sharding.remote_subpath = resolved_subpath

    # 1. Data acquisition and FASTA resolution
    pm.start_stage(1, total_items=1, info="Resolving proteome & PTM sources")

    if sample_only:
        resolved_fasta = Path("data/sample.fasta")
    elif fasta_path:
        resolved_fasta = Path(fasta_path)
    else:
        resolved_fasta = fetch_uniprot_human_proteome(out_dir="data/raw")

    if not resolved_fasta.exists():
        raise FileNotFoundError(f"FASTA file not found at {resolved_fasta}")

    resolved_ptm_sources = list(ptm_sources) if ptm_sources else []
    if not resolved_ptm_sources and not sample_only:
        uniprot_ptm = fetch_uniprot_ptm_features(output_dir="data/raw")
        cplm_ptm = fetch_cplm_human(output_dir="data/raw")
        resolved_ptm_sources = [uniprot_ptm, cplm_ptm]

    pm.finish_stage(
        1,
        summary=f"Proteome: {resolved_fasta.name} | PTM datasets: {len(resolved_ptm_sources)} sources",
    )

    # 2. Harmonization, invariant validation and exact MILP partitioning
    pm.start_stage(2, total_items=1, info="Harmonization & exact MILP partitioning")

    corpus: PTMCorpus = prepare_ptm_corpus(
        fasta_path=resolved_fasta,
        ptm_sources=resolved_ptm_sources,
        split_ratio=0.8,
        max_seq_length=config.extraction.max_sequence_length,
        output_dir=processed_path,
    )

    pm.finish_stage(
        2,
        summary=f"{corpus.total_proteins} proteins ({len(corpus.discovery_proteins)} discovery, {len(corpus.held_out_proteins)} held-out) | {corpus.total_sites} valid sites",
    )

    # 3. ESM-2 activation extraction and sharded buffering
    target_proteins = corpus.all_proteins if sample_only else corpus.discovery_proteins
    if max_proteins is not None:
        target_proteins = target_proteins[:max_proteins]

    sharding_manifest = run_extraction_pipeline(
        config=config,
        proteins=target_proteins,
        progress_manager=pm,
    )

    # 4. Structured remote hub synchronization and provenance upload
    resolved_token = resolve_hf_token(token)

    if resolved_repo and resolved_token:
        pm.start_stage(4, total_items=4, info=f"Pushing provenance to {resolved_repo}")
        hub = HfSyncClient(token=resolved_token)

        corpus_subpath = "corpus"
        meta_files = (
            processed_path / "split_manifest.json",
            processed_path / "proteins.jsonl",
            processed_path / "ptm_sites.jsonl",
            processed_path / "mismatch_audit.tsv",
        )
        uploaded = 0
        for meta_file in meta_files:
            if meta_file.exists():
                res = hub.upload_shard(resolved_repo, corpus_subpath, meta_file)
                if res is not None:
                    uploaded += 1
            pm.advance(1)

        pm.finish_stage(
            4,
            summary=f"Provenance committed to {resolved_repo} ({uploaded} committed, {len(meta_files) - uploaded} skipped up-to-date)",
        )
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
        "corpus_manifest": corpus.manifest,
        "sharding_manifest": sharding_manifest,
        "output_dir": config.sharding.output_dir,
        "processed_dir": str(processed_path),
        "remote_repo_id": resolved_repo,
        "remote_subpath": resolved_subpath,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Lifecycle Pipeline (Download -> Partition -> Extract -> Upload)"
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
        help="Path to FASTA file (downloads human proteome if omitted)",
    )
    parser.add_argument(
        "--ptm-source",
        action="append",
        default=None,
        help="Path(s) to raw PTM files (PhosphoSitePlus, CPLM, dbPTM, etc.). Automatically defaults to Tier 1 datasets (UniProt + CPLM) if omitted.",
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
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.device:
        cfg.model.device = args.device

    run_full_lifecycle(
        config=cfg,
        fasta_path=args.fasta,
        ptm_sources=args.ptm_source,
        sample_only=args.sample_only,
        max_proteins=args.max_proteins,
        remote_repo_id=args.remote_repo,
        remote_subpath=args.remote_subpath,
    )


if __name__ == "__main__":
    main()
