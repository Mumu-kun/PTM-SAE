"""Unified end-to-end lifecycle pipeline: Download -> Harmonize & Partition -> Extract -> Upload -> Verify."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ptm_sae.config import PipelineConfig
from ptm_sae.data import PTMCorpus, prepare_ptm_corpus
from ptm_sae.data.fetcher import (
    fetch_cplm_human,
    fetch_uniprot_human_proteome,
    fetch_uniprot_ptm_features,
)
from ptm_sae.extraction.hub import HfSyncClient, resolve_hf_token
from ptm_sae.extraction.pipeline import run_extraction_pipeline
from ptm_sae.extraction.reader import SafeTensorsReader


def run_full_lifecycle(
    config: PipelineConfig,
    fasta_path: str | Path | None = None,
    ptm_sources: Sequence[str | Path] | None = None,
    processed_dir: str | Path = "data/processed",
    sample_only: bool = False,
    max_proteins: int | None = None,
    remote_repo_id: str | None = None,
    remote_subpath: str | None = None,
    token: str | None = None,
) -> dict:
    """
    Executes the entire data acquisition, extraction, and remote synchronization lifecycle:
    1. Download: Streams reviewed human proteome from UniProt REST API if no FASTA is given.
    2. Harmonization & Splitting: Validates biological invariants and partitions via exact MILP.
    3. Activation Extraction: Extracts residue-level tensors with automatic CUDA OOM fallback.
    4. Remote Upload: Pushes shards, auxiliary context, and manifests to structured HF Hub subpaths.
    5. Zero-Copy Readback Verification: Performs sanity verification on output SafeTensors.
    """
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. Resolve structured remote paths
    resolved_repo = remote_repo_id or config.sharding.remote_repo_id
    resolved_subpath = remote_subpath or config.resolve_remote_subpath()

    config.sharding.remote_repo_id = resolved_repo
    config.sharding.remote_subpath = resolved_subpath

    # 1. Data acquisition and FASTA resolution
    print("\n[Stage 1] Data Acquisition & FASTA Resolution")

    if sample_only:
        resolved_fasta = Path("data/sample.fasta")
        print(f"[Stage 1] Using sample FASTA: {resolved_fasta}")
    elif fasta_path:
        resolved_fasta = Path(fasta_path)
        print(f"[Stage 1] Using provided FASTA: {resolved_fasta}")
    else:
        print(
            "[Stage 1] No FASTA provided: streaming reviewed canonical human proteome from UniProt..."
        )
        resolved_fasta = fetch_uniprot_human_proteome(out_dir="data/raw")
        print(f"[Stage 1] UniProt human proteome cached at: {resolved_fasta}")

    if not resolved_fasta.exists():
        raise FileNotFoundError(f"FASTA file not found at {resolved_fasta}")

    # Resolve PTM sources: user provided or automatic Tier 1 acquisition (UniProt + CPLM)
    resolved_ptm_sources = list(ptm_sources) if ptm_sources else []
    if not resolved_ptm_sources and not sample_only:
        print(
            "[Stage 1] No PTM source provided: acquiring Tier 1 datasets (UniProt features + CPLM 4.0)..."
        )
        uniprot_ptm = fetch_uniprot_ptm_features(output_dir="data/raw")
        cplm_ptm = fetch_cplm_human(output_dir="data/raw")
        resolved_ptm_sources = [uniprot_ptm, cplm_ptm]
        print(
            f"[Stage 1] Tier 1 PTM sources ready: {uniprot_ptm.name}, {cplm_ptm.name}"
        )

    # 2. Harmonization, invariant validation and exact MILP partitioning
    print("\n[Stage 2] Harmonization, Invariant Validation & Exact MILP Partitioning")

    corpus: PTMCorpus = prepare_ptm_corpus(
        fasta_path=resolved_fasta,
        ptm_sources=resolved_ptm_sources,
        split_ratio=0.8,
        max_seq_length=config.extraction.max_sequence_length,
        output_dir=processed_path,
    )
    print(
        f"[Stage 2] Corpus assembled: {corpus.total_proteins} proteins ({len(corpus.discovery_proteins)} discovery, {len(corpus.held_out_proteins)} held-out)."
    )
    print(
        f"[Stage 2] Validated PTM sites: {corpus.total_sites} | Mismatches dropped: {len(corpus.audit_log)}"
    )

    # 3. ESM-2 activation extraction and sharded buffering
    print("\n[Stage 3] ESM-2 Activation Extraction & Sharded Buffering")

    # In production, extract the discovery partition; in sample mode, extract all
    target_proteins = corpus.all_proteins if sample_only else corpus.discovery_proteins
    if max_proteins is not None:
        target_proteins = target_proteins[:max_proteins]
        print(
            f"[Stage 3] Capped extraction set to first {max_proteins} proteins (--max-proteins)."
        )
    else:
        print(f"[Stage 3] Target extraction set: {len(target_proteins)} proteins.")

    sharding_manifest = run_extraction_pipeline(
        config=config,
        proteins=target_proteins,
    )

    # 4. Structured remote hub synchronization and provenance upload
    print("\n[Stage 4] Structured Remote Hub Synchronization & Provenance Upload")

    resolved_token = resolve_hf_token(token)

    if resolved_repo and resolved_token:
        print(
            f"[Stage 4] Synchronizing artifacts to Hugging Face Hub ({resolved_repo})..."
        )
        hub = HfSyncClient(token=resolved_token)

        # 4a. Upload corpus split provenance to dedicated 'corpus/' namespace
        corpus_subpath = "corpus"
        for meta_file in (
            processed_path / "split_manifest.json",
            processed_path / "proteins.jsonl",
            processed_path / "ptm_sites.jsonl",
            processed_path / "mismatch_audit.tsv",
        ):
            if meta_file.exists():
                hub.upload_shard(resolved_repo, corpus_subpath, meta_file)
                print(f"[Stage 4] Uploaded to {corpus_subpath}/: {meta_file.name}")

        print(f"[Stage 4] Shards and embeddings committed under: {resolved_subpath}/")
        print("[Stage 4] Remote storage synchronization complete.")
    elif resolved_repo and not resolved_token:
        print(
            f"[Stage 4] Warning: remote_repo_id ('{resolved_repo}') is set, but no HF_TOKEN was found. Remote upload skipped."
        )
    else:
        print(
            "[Stage 4] Local Mode: remote_repo_id is not set. All shards and manifests retained in local cache."
        )

    # 5. Zero-copy readback sanity verification
    print("\n[Stage 5] Zero-Copy Readback Sanity Verification")

    reader = SafeTensorsReader(
        cache_dir=config.sharding.output_dir,
        remote_repo_id=resolved_repo,
        remote_subpath=resolved_subpath,
        token=resolved_token,
    )
    available_proteins = reader.list_proteins()
    print(
        f"[Stage 5] Reader verified: {len(available_proteins)} proteins indexed in SafeTensors manifest."
    )

    if available_proteins:
        probe_id = available_proteins[0]
        acts = reader.get_protein_activations(probe_id)
        res_1 = reader.get_residue_activation(probe_id, 1)
        mean_vec = reader.get_mean_pooled_embedding(probe_id)
        print(f"[Stage 5] Probe protein '{probe_id}':")
        print(f"         - Residue activations tensor: shape {acts.shape}")
        print(f"         - Biological residue 1 vector: shape {res_1.shape}")
        print(f"         - Global mean-pooled embedding: shape {mean_vec.shape}")

    print("\n" + "=" * 70)
    print("LIFECYCLE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

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
