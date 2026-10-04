"""FASTA ingestion and residue-stratum lookup.

PTM-type canonicalization and the per-source parsers/reader registry that used to live here
(PhosphoSitePlus, dbPTM, UniProt-features, CPLM) are retired -- superseded by
`ptm_sae.corpus.labels`'s chemistry-validated cascade, ported from the partner's N1 pipeline
(see the implementation plan, decision 5). PhosphoSitePlus itself is deliberately not carried
forward: N1 already gets robust phosphorylation coverage from license-clean bulk-downloadable
sources (CPLM/dbPTM/qPTM).

TODO(Stage 2): `data/__init__.py::prepare_ptm_corpus` still calls `read_ptm_sites`, which no
longer exists. That caller (and `PTMCorpus`/`solve_milp_partition` alongside it) are retired
wholesale in Stage 2 of the plan, not patched here.
"""

import os
import re
from pathlib import Path

from ptm_sae.data.schema import Protein

# Canonical chemical strata mapping -- codes match ptm_sae.corpus.config.Config.stratum_residues
# exactly (K, ST, N, C, R, Y, ...), not full English words. Do not translate these.
STRATUM_MAPPING: dict[str, str] = {
    "S": "ST",
    "T": "ST",
    "K": "K",
    "N": "N",
    "C": "C",
    "R": "R",
    "Y": "Y",
}


def get_stratum_for_residue(residue: str) -> str:
    """Return canonical chemical stratum code for an amino acid code."""
    return STRATUM_MAPPING.get(residue.upper(), "other")


def parse_uniprot_fasta(
    fasta_path_or_text: str | Path,
    max_sequence_length: int = 1022,
) -> tuple[list[Protein], list[tuple[str, int]]]:
    """
    Parse Swiss-Prot / UniProt FASTA into canonical Protein objects.
    Enforces maximum sequence length limit (<= 1022).
    """
    valid: list[Protein] = []
    skipped: list[tuple[str, int]] = []

    current_header = None
    current_seq_parts: list[str] = []

    def flush():
        nonlocal current_header, current_seq_parts
        if current_header is None:
            return

        seq = "".join(current_seq_parts).strip().upper().replace("*", "")
        match = re.search(r">[a-zA-Z0-9_-]+\|([a-zA-Z0-9_-]+)\|", current_header)
        u_id = match.group(1) if match else current_header[1:].split()[0]

        if len(seq) > max_sequence_length:
            skipped.append((u_id, len(seq)))
        elif len(seq) > 0:
            valid.append(
                Protein(
                    uniprot_id=u_id,
                    sequence=seq,
                    length=len(seq),
                    reviewed=True,
                    taxonomy_id=9606,
                    header=current_header,
                )
            )

        current_header = None
        current_seq_parts = []

    def _consume(line_stream) -> None:
        nonlocal current_header
        for line in line_stream:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                flush()
                current_header = line
            else:
                current_seq_parts.append(line)
        flush()

    if isinstance(fasta_path_or_text, Path) or (
        isinstance(fasta_path_or_text, str) and os.path.exists(fasta_path_or_text)
    ):
        with open(fasta_path_or_text, encoding="utf-8") as file_handle:
            _consume(file_handle)
    else:
        _consume(fasta_path_or_text.splitlines())

    return valid, skipped
