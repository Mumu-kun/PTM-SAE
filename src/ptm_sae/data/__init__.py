"""Data acquisition, harmonization, and splitting subsystem."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ptm_sae.data.fetcher import (
    fetch_cplm_human,
    fetch_uniprot_human_proteome,
    fetch_uniprot_ptm_features,
    harmonize_and_validate_ptm_sites,
)
from ptm_sae.data.ingestion import (
    canonicalize_ptm_type,
    get_stratum_for_residue,
    parse_uniprot_fasta,
    read_ptm_sites,
    register_reader,
)
from ptm_sae.data.schema import (
    Protein,
    PTMObservation,
    UnifiedResidueSite,
)
from ptm_sae.data.splitting import (
    build_cluster_profiles,
    partition_dataset,
    solve_greedy_partition,
    solve_milp_partition,
)


@dataclass
class PTMCorpus:
    """Consolidated corpus representation ready for downstream extraction and SAE training."""

    discovery_proteins: list[Protein]
    held_out_proteins: list[Protein]
    all_proteins: list[Protein]
    sites: list[UnifiedResidueSite]
    manifest: dict[str, Any]
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    discovery_train_proteins: list[Protein] = field(default_factory=list)
    discovery_val_proteins: list[Protein] = field(default_factory=list)

    @property
    def total_proteins(self) -> int:
        return len(self.all_proteins)

    @property
    def total_sites(self) -> int:
        return len(self.sites)


def prepare_ptm_corpus(
    fasta_path: str | Path,
    ptm_sources: Sequence[str | Path] | None = None,
    split_ratio: float = 0.8,
    max_seq_length: int = 1022,
    cluster_mapping: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> PTMCorpus:
    """
    Deep entry point for data acquisition, harmonization, and splitting.
    Orchestrates parsing, validation, multi-label aggregation, MILP partitioning, and export.
    """
    # 1. Parse FASTA and enforce sequence length constraint
    valid_proteins, skipped = parse_uniprot_fasta(
        fasta_path, max_sequence_length=max_seq_length
    )
    protein_map = {p.uniprot_id: p for p in valid_proteins}

    # 2. Ingest raw PTM observations across all registered sources
    observations: list[PTMObservation] = []
    if ptm_sources:
        for src in ptm_sources:
            src_path = Path(src)
            if src_path.exists():
                observations.extend(read_ptm_sites(src_path))

    # 3. Enforce sequence invariant (sequence[pos - 1] == residue) and aggregate crosstalk
    sites, audit_log = harmonize_and_validate_ptm_sites(observations, protein_map)

    # 4. Partition clusters into non-homologous sets via exact MILP (HiGHS)
    preprocessed_proteins, cluster_assignments = partition_dataset(
        proteins=valid_proteins,
        sites=sites,
        cluster_mapping=cluster_mapping,
        target_discovery_ratio=split_ratio,
    )

    discovery = [
        p
        for p in preprocessed_proteins
        if p.partition in ("discovery", "discovery_train", "discovery_val")
    ]
    discovery_train = [
        p for p in preprocessed_proteins if p.partition == "discovery_train"
    ]
    discovery_val = [p for p in preprocessed_proteins if p.partition == "discovery_val"]
    held_out = [p for p in preprocessed_proteins if p.partition == "held_out"]

    # 5. Compute token and partition distribution summary
    total_tokens = sum(p.length for p in preprocessed_proteins)
    discovery_tokens = sum(p.length for p in discovery)
    discovery_train_tokens = sum(p.length for p in discovery_train)
    discovery_val_tokens = sum(p.length for p in discovery_val)
    held_out_tokens = sum(p.length for p in held_out)

    manifest = {
        "total_proteins": len(preprocessed_proteins),
        "skipped_length_exceeded": len(skipped),
        "total_tokens": total_tokens,
        "discovery_proteins": len(discovery),
        "discovery_tokens": discovery_tokens,
        "discovery_token_ratio": discovery_tokens / max(1, total_tokens),
        "discovery_train_proteins": len(discovery_train),
        "discovery_train_tokens": discovery_train_tokens,
        "discovery_train_token_ratio": discovery_train_tokens / max(1, total_tokens),
        "discovery_val_proteins": len(discovery_val),
        "discovery_val_tokens": discovery_val_tokens,
        "discovery_val_token_ratio": discovery_val_tokens / max(1, total_tokens),
        "held_out_proteins": len(held_out),
        "held_out_tokens": held_out_tokens,
        "held_out_token_ratio": held_out_tokens / max(1, total_tokens),
        "total_sites": len(sites),
        "total_audit_mismatches": len(audit_log),
        "cluster_count": len(cluster_assignments),
    }

    # 6. Optional persistence to processed directory
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Write partitioned proteins JSONL
        with open(out_path / "proteins.jsonl", "w", encoding="utf-8") as f:
            f.writelines(p.model_dump_json() + "\n" for p in preprocessed_proteins)

        # Write validated PTM sites JSONL
        with open(out_path / "ptm_sites.jsonl", "w", encoding="utf-8") as f:
            f.writelines(s.model_dump_json() + "\n" for s in sites)

        # Write audit manifest
        with open(out_path / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Write coordinate mismatch audit report if any dropped
        if audit_log:
            with open(out_path / "mismatch_audit.tsv", "w", encoding="utf-8") as f:
                f.write(
                    "uniprot_id\tposition\texpected\tactual\treason\tsource_db\tptm_type\n"
                )
                f.writelines(
                    f"{row.get('uniprot_id')}\t{row.get('position')}\t{row.get('expected_residue')}\t{row.get('actual_residue')}\t{row.get('reason')}\t{row.get('source_db')}\t{row.get('ptm_type')}\n"
                    for row in audit_log
                )

    return PTMCorpus(
        discovery_proteins=discovery,
        held_out_proteins=held_out,
        all_proteins=preprocessed_proteins,
        sites=sites,
        manifest=manifest,
        audit_log=audit_log,
        discovery_train_proteins=discovery_train,
        discovery_val_proteins=discovery_val,
    )


__all__ = [
    "PTMCorpus",
    "PTMObservation",
    "Protein",
    "UnifiedResidueSite",
    "canonicalize_ptm_type",
    "fetch_cplm_human",
    "fetch_uniprot_human_proteome",
    "fetch_uniprot_ptm_features",
    "get_stratum_for_residue",
    "harmonize_and_validate_ptm_sites",
    "parse_uniprot_fasta",
    "prepare_ptm_corpus",
    "read_ptm_sites",
    "register_reader",
]
