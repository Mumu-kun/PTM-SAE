"""Canonical domain schema for protein sequences and PTM annotations."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Protein(BaseModel):
    """Canonical sequence representation with cluster and partition assignments."""

    uniprot_id: str
    sequence: str
    length: int
    cluster_id: str = ""
    partition: Literal["discovery", "discovery_train", "discovery_val", "held_out"] = (
        "discovery"
    )
    has_annotated_ptm: bool = False
    stratum_counts: dict[str, int] = Field(default_factory=dict)
    reviewed: bool = True
    taxonomy_id: int = 9606
    header: str = ""


class PTMObservation(BaseModel):
    """Atomic PTM site record parsed from an individual data source."""

    source_db: str
    uniprot_id: str
    position: int  # 1-indexed biological coordinate
    residue: str  # Single-letter amino acid code
    canonical_ptm_type: str
    raw_ptm_name: str = ""
    evidence_tier: Literal["experimental", "curated", "predicted"] = "experimental"


class UnifiedResidueSite(BaseModel):
    """
    Physical residue coordinate with multi-label PTM set.
    Accommodates PTM crosstalk (multiple modifications on the same physical residue).
    """

    uniprot_id: str
    position: int  # 1-indexed biological coordinate
    residue: str  # Single-letter amino acid code
    stratum: str  # Chemical stratum ('lysine', 'serine_threonine', etc.)
    ptm_types: set[str] = Field(default_factory=set)
    sources: set[str] = Field(default_factory=set)
    source_multiplicity: dict[str, int] = Field(default_factory=dict)
    is_multi_label: bool = False
    negative_tier: Literal["verified", "hard", "background"] | None = None
    partition: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
