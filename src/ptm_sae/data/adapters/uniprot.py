"""Adapters for UniProt FASTA and Feature tables."""
from pathlib import Path
import re
from typing import Iterator, List, Tuple, Union
from ptm_sae.data.adapters.base import BaseSiteAdapter, canonicalize_ptm_type
from ptm_sae.data.schema import PTMObservation, ProteinSequence


def parse_uniprot_fasta(
    fasta_path_or_text: Union[str, Path],
    max_sequence_length: int = 1022,
) -> Tuple[List[ProteinSequence], List[Tuple[str, int]]]:
    """
    Parse Swiss-Prot / UniProt FASTA into canonical ProteinSequence objects.
    Enforces maximum sequence length limit (<= 1022).
    """
    valid: List[ProteinSequence] = []
    skipped: List[Tuple[str, int]] = []

    current_header = None
    current_seq_parts: List[str] = []

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
            valid.append(ProteinSequence(
                uniprot_id=u_id,
                sequence=seq,
                length=len(seq),
                reviewed=True,
                taxonomy_id=9606,
                header=current_header,
            ))
        current_header = None
        current_seq_parts = []

    if isinstance(fasta_path_or_text, Path) or (isinstance(fasta_path_or_text, str) and Path(fasta_path_or_text).exists()):
        with open(fasta_path_or_text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush()
                    current_header = line
                else:
                    current_seq_parts.append(line)
            flush()
    else:
        for line in fasta_path_or_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line
            else:
                current_seq_parts.append(line)
        flush()

    return valid, skipped


class UniProtFeatureTsvAdapter(BaseSiteAdapter):
    """
    Parses exported UniProt Feature tables (TSV) containing columns:
    Entry, Feature key, Position(s), Description
    """

    def can_handle(self, source: Union[str, Path]) -> bool:
        src_str = str(source).lower()
        return "uniprot" in src_str and (src_str.endswith(".tsv") or src_str.endswith(".tab"))

    def parse(self, source: Union[str, Path]) -> Iterator[PTMObservation]:
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                return
            headers = [h.strip().lower() for h in header_line.split("\t")]
            
            entry_idx = headers.index("entry") if "entry" in headers else 0
            key_idx = headers.index("feature key") if "feature key" in headers else 1
            pos_idx = headers.index("position(s)") if "position(s)" in headers else 2
            desc_idx = headers.index("description") if "description" in headers else 3

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) <= max(entry_idx, key_idx, pos_idx):
                    continue
                feature_key = parts[key_idx].strip().upper()
                if feature_key not in ("MOD_RES", "CARBOHYD", "LIPID"):
                    continue

                u_id = parts[entry_idx].strip()
                pos_str = parts[pos_idx].strip()
                desc = parts[desc_idx].strip() if len(parts) > desc_idx else feature_key

                # Parse single integer position (skip ranges like '1..20' unless single site)
                match = re.search(r"^(\d+)$", pos_str)
                if not match:
                    continue
                pos = int(match.group(1))

                # Extract residue from description if present (e.g. 'Phosphoserine', 'N-acetyllysine')
                residue = "S"
                if "threonine" in desc.lower():
                    residue = "T"
                elif "tyrosine" in desc.lower():
                    residue = "Y"
                elif "lysine" in desc.lower():
                    residue = "K"
                elif "asparagine" in desc.lower():
                    residue = "N"
                elif "cysteine" in desc.lower():
                    residue = "C"

                yield PTMObservation(
                    source_db="UniProt",
                    uniprot_id=u_id,
                    position=pos,
                    residue=residue,
                    canonical_ptm_type=canonicalize_ptm_type(desc),
                    raw_ptm_name=desc,
                    evidence_tier="curated",
                )
