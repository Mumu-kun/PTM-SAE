"""Base abstract adapter definitions and canonical PTM mapping."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from ptm_sae.data.schema import PTMObservation, ProteinSequence

# Canonical chemical strata
STRATUM_MAPPING: Dict[str, str] = {
    "S": "serine_threonine",
    "T": "serine_threonine",
    "K": "lysine",
    "N": "asparagine",
    "C": "cysteine",
    "R": "arginine",
    "Y": "tyrosine",
}


def get_stratum_for_residue(residue: str) -> str:
    """Returns canonical chemical stratum for an amino acid, or 'other'."""
    return STRATUM_MAPPING.get(residue.upper(), "other")


# Standard PTM category normalization dictionary
CANONICAL_PTM_SYNONYMS: Dict[str, str] = {
    "phosphorylation": "Phosphorylation",
    "phospho": "Phosphorylation",
    "phosphoserine": "Phosphorylation",
    "phosphothreonine": "Phosphorylation",
    "phosphotyrosine": "Phosphorylation",
    "p": "Phosphorylation",
    "ubiquitination": "Ubiquitination",
    "ubiquitin": "Ubiquitination",
    "ub": "Ubiquitination",
    "glycine glycyl": "Ubiquitination",
    "acetylation": "Acetylation",
    "acetyl": "Acetylation",
    "ac": "Acetylation",
    "n6-acetyllysine": "Acetylation",
    "methylation": "Methylation",
    "methyl": "Methylation",
    "me": "Methylation",
    "mono-methylation": "Methylation",
    "di-methylation": "Methylation",
    "tri-methylation": "Methylation",
    "n-linked glycosylation": "N-Glycosylation",
    "n-glycosylation": "N-Glycosylation",
    "o-linked glycosylation": "O-Glycosylation",
    "o-glycosylation": "O-Glycosylation",
    "o-glcnac": "O-Glycosylation",
    "o-glcnacylation": "O-Glycosylation",
    "sumoylation": "Sumoylation",
    "sumo": "Sumoylation",
}


def canonicalize_ptm_type(raw_type: str) -> str:
    """Normalize diverse database modification strings into canonical names."""
    cleaned = raw_type.strip().lower()
    for key, canonical in CANONICAL_PTM_SYNONYMS.items():
        if key in cleaned:
            return canonical
    return raw_type.strip().title()


class BaseSiteAdapter(ABC):
    """Abstract base class for parsing PTM site annotations from diverse sources."""

    @abstractmethod
    def can_handle(self, source: Union[str, Path]) -> bool:
        """Check if this adapter can handle the given file or source string."""
        pass

    @abstractmethod
    def parse(self, source: Union[str, Path]) -> Iterator[PTMObservation]:
        """Parse source data into an iterator of canonical PTMObservation objects."""
        pass
