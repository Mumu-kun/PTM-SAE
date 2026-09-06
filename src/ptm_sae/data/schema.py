"""Canonical domain schema for protein sequences and PTM annotations."""
from typing import Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field


class ProteinSequence(BaseModel):
    """Canonical sequence representation."""
    uniprot_id: str
    sequence: str
    length: int
    reviewed: bool = True
    taxonomy_id: int = 9606
    header: str = ""


class PreprocessedProtein(ProteinSequence):
    """Protein annotated with corpus subsets, stratum counts, and partition assignments."""
    has_annotated_ptm: bool = False
    stratum_counts: Dict[str, int] = Field(default_factory=dict)
    cluster_id: str = ""
    partition: Literal["discovery", "held_out"] = "discovery"


class PTMObservation(BaseModel):
    """Atomic PTM site record parsed directly from an individual database."""
    source_db: str
    uniprot_id: str
    position: int  # Strictly 1-indexed biological coordinate
    residue: str   # Single-letter amino acid code
    canonical_ptm_type: str
    raw_ptm_name: str = ""
    evidence_tier: Literal["experimental", "curated", "predicted"] = "experimental"
    flanking_15mer: Optional[str] = None
    upstream_enzyme: Optional[str] = None
    pubmed_ids: List[str] = Field(default_factory=list)


class UnifiedResidueSite(BaseModel):
    """
    Unified multi-label representation per residue coordinate.
    Accommodates PTM crosstalk (multiple modifications on the same physical residue).
    """
    uniprot_id: str
    position: int  # 1-indexed
    residue: str
    stratum: str
    flanking_15mer: str = ""

    # Multi-label modification types
    ptm_types: Set[str] = Field(default_factory=set)
    is_modified: bool = False
    is_multi_label: bool = False

    # Evidence & Quality metrics per PTM type
    evidence_tiers: Dict[str, str] = Field(default_factory=dict)
    source_multiplicity: Dict[str, int] = Field(default_factory=dict)
    ambiguity_mask: Dict[str, bool] = Field(default_factory=dict)

    # Negative tier classification
    negative_tier: Optional[Literal["verified", "hard", "background"]] = "background"

    # Sparse metadata
    upstream_enzymes: Dict[str, List[str]] = Field(default_factory=dict)


class AcquisitionManifestEntry(BaseModel):
    """Cryptographic provenance record for a fetched data artifact."""
    source_name: str
    url_or_path: str
    sha256: str
    file_size_bytes: int
    record_count: int
    timestamp_utc: str
