"""Canonical domain schema for protein sequences and PTM annotations."""

from typing import Literal

from pydantic import BaseModel, Field


class Protein(BaseModel):
    """Canonical sequence representation with cluster and partition assignments."""

    uniprot_id: str
    sequence: str
    length: int
    cluster_id: str = ""
    partition: Literal["discovery_train", "discovery_val", "held_out"] = (
        "discovery_train"
    )
    has_annotated_ptm: bool = False
    stratum_counts: dict[str, int] = Field(default_factory=dict)
    reviewed: bool = True
    taxonomy_id: int = 9606
    header: str = ""
