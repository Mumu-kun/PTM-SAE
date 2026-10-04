"""Data acquisition and canonical schema subsystem.

Corpus/label preparation itself (acquisition, CD-HIT clustering, the PTM label cascade) now
lives in `ptm_sae.corpus`, ported from the partner's N1 pipeline (implementation plan, Stage 1).
This package keeps only what still applies downstream of that: raw sequence fetching
(`fetch_uniprot_human_proteome`), FASTA parsing, and the shared `Protein`/`PTMObservation`/
`UnifiedResidueSite` schema.
"""

from ptm_sae.data.fetcher import fetch_uniprot_human_proteome
from ptm_sae.data.ingestion import (
    get_stratum_for_residue,
    parse_uniprot_fasta,
)
from ptm_sae.data.schema import (
    Protein,
    PTMObservation,
    UnifiedResidueSite,
)

__all__ = [
    "PTMObservation",
    "Protein",
    "UnifiedResidueSite",
    "fetch_uniprot_human_proteome",
    "get_stratum_for_residue",
    "parse_uniprot_fasta",
]
