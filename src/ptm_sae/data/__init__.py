"""Data acquisition and canonical schema subsystem.

Corpus/label preparation itself (acquisition, CD-HIT clustering, the PTM label cascade) was ported
from the partner's N1 pipeline and is now archived on git branch `archive/corpus-build`; `ptm_sae.corpus`
keeps only the maintenance side (partition audit, verify, resplit, upload).
This package keeps only what still applies downstream of that: raw sequence fetching
(`fetch_uniprot_human_proteome`), FASTA parsing, and the shared `Protein` schema.
"""

from ptm_sae.data.fetcher import fetch_uniprot_human_proteome
from ptm_sae.data.ingestion import (
    get_stratum_for_residue,
    parse_uniprot_fasta,
)
from ptm_sae.data.schema import Protein

__all__ = [
    "Protein",
    "fetch_uniprot_human_proteome",
    "get_stratum_for_residue",
    "parse_uniprot_fasta",
]
