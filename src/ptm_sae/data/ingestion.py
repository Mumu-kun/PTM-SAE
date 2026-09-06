"""Deep ingestion module with lightweight reader registry for multi-source PTM datasets."""
from collections.abc import Callable
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from ptm_sae.data.schema import PTMObservation, Protein

# Canonical chemical strata mapping
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
    """Return canonical chemical stratum for an amino acid code."""
    return STRATUM_MAPPING.get(residue.upper(), "other")


# Regex patterns for canonical PTM normalization (ordered: specific patterns before general)
CANONICAL_PTM_PATTERNS: List[Tuple[str, str]] = [
    # O-Glycosylation & O-GlcNAc (before generic acetylation)
    (r"\bo-glcnac", "O-Glycosylation"),
    (r"\bo-linked glycosylation", "O-Glycosylation"),
    (r"\bo-glycosylation", "O-Glycosylation"),
    # N-Glycosylation
    (r"\bn-linked glycosylation", "N-Glycosylation"),
    (r"\bn-glycosylation", "N-Glycosylation"),
    # Phosphorylation
    (r"phospho", "Phosphorylation"),
    (r"-p\b", "Phosphorylation"),
    # Acetylation
    (r"acetyl", "Acetylation"),
    (r"-ac\b", "Acetylation"),
    # Ubiquitination
    (r"ubiquitin", "Ubiquitination"),
    (r"glycine glycyl", "Ubiquitination"),
    (r"-ub\b", "Ubiquitination"),
    # Methylation
    (r"methyl", "Methylation"),
    (r"-m\b", "Methylation"),
    # Sumoylation
    (r"sumo", "Sumoylation"),
    (r"-su\b", "Sumoylation"),
]


def canonicalize_ptm_type(raw_type: str) -> str:
    """Normalize raw PTM modification strings into canonical names."""
    cleaned = raw_type.strip().lower()
    for pattern, canonical in CANONICAL_PTM_PATTERNS:
        if re.search(pattern, cleaned):
            return canonical
    return raw_type.strip().title()


def parse_uniprot_fasta(
    fasta_path_or_text: Union[str, Path],
    max_sequence_length: int = 1022,
) -> Tuple[List[Protein], List[Tuple[str, int]]]:
    """
    Parse Swiss-Prot / UniProt FASTA into canonical Protein objects.
    Enforces maximum sequence length limit (<= 1022).
    """
    valid: List[Protein] = []
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
            valid.append(Protein(
                uniprot_id=u_id,
                sequence=seq,
                length=len(seq),
                reviewed=True,
                taxonomy_id=9606,
                header=current_header,
            ))

        current_header = None
        current_seq_parts = []

    # Stream lines from file path or raw string
    lines = (
        open(fasta_path_or_text, "r", encoding="utf-8")
        if isinstance(fasta_path_or_text, Path) or (isinstance(fasta_path_or_text, str) and Path(fasta_path_or_text).exists())
        else fasta_path_or_text.splitlines()
    )

    try:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                flush()
                current_header = line
            else:
                current_seq_parts.append(line)
        flush()
    finally:
        if hasattr(lines, "close"):
            lines.close()

    return valid, skipped


# Reader Registry
ReaderFunc = Callable[[Path, Any], Iterator[PTMObservation]]
_READER_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_reader(
    name: str,
    header_keywords: Sequence[str] = (),
    file_keywords: Sequence[str] = (),
) -> Callable[[ReaderFunc], ReaderFunc]:
    """
    Decorator registering a specialized PTM site reader with zero multi-file boilerplate.
    """
    def decorator(fn: ReaderFunc) -> ReaderFunc:
        _READER_REGISTRY[name.lower()] = {
            "func": fn,
            "header_keywords": [k.lower() for k in header_keywords],
            "file_keywords": [k.lower() for k in file_keywords],
        }
        return fn
    return decorator


@register_reader(
    name="phosphositeplus",
    header_keywords=["mod_rsd", "site_grp_id"],
    file_keywords=["phosphosite", "site_dataset"],
)
def _parse_phosphositeplus(path: Path, **kwargs: Any) -> Iterator[PTMObservation]:
    """Parses PhosphoSitePlus export datasets (e.g. Phosphorylation_site_dataset.txt)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        headers: Optional[List[str]] = None

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse header on first non-comment line
            if headers is None:
                headers = [h.strip().upper() for h in line.split("\t")]
                acc_idx = headers.index("ACC_ID") if "ACC_ID" in headers else 1
                mod_idx = headers.index("MOD_RSD") if "MOD_RSD" in headers else 4
                org_idx = headers.index("ORGANISM") if "ORGANISM" in headers else 6
                max_idx = max(acc_idx, mod_idx, org_idx)
                continue

            parts = line.split("\t")
            if len(parts) <= max_idx or parts[org_idx].strip().lower() != "human":
                continue

            u_id = parts[acc_idx].strip()
            mod_rsd = parts[mod_idx].strip()  # e.g. 'S15-p' or 'K382-ub'

            match = re.search(r"^([A-Z])(\d+)", mod_rsd)
            if not match:
                continue

            yield PTMObservation(
                source_db="PhosphoSitePlus",
                uniprot_id=u_id,
                position=int(match.group(2)),
                residue=match.group(1),
                canonical_ptm_type=canonicalize_ptm_type(mod_rsd),
                raw_ptm_name=mod_rsd,
                evidence_tier="experimental",
            )


@register_reader(
    name="dbptm",
    header_keywords=["uniprot_id", "ptm_type"],
    file_keywords=["dbptm"],
)
def _parse_dbptm(path: Path, **kwargs: Any) -> Iterator[PTMObservation]:
    """Parses dbPTM tab-delimited site export tables."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
        if not header_line:
            return
        headers = [h.strip().lower() for h in header_line.split("\t")]

        id_col = next((i for i, h in enumerate(headers) if "id" in h or "acc" in h), 0)
        ptm_col = next((i for i, h in enumerate(headers) if "ptm" in h or "type" in h), 2)
        pos_col = next((i for i, h in enumerate(headers) if "pos" in h or "site" in h), 3)
        flank_col = next((i for i, h in enumerate(headers) if "flank" in h or "seq" in h or "window" in h), None)
        max_idx = max(id_col, ptm_col, pos_col)

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max_idx:
                continue

            try:
                pos = int(parts[pos_col].strip())
            except ValueError:
                continue

            u_id = parts[id_col].strip()
            ptm_raw = parts[ptm_col].strip()
            flank = parts[flank_col].strip() if flank_col is not None and len(parts) > flank_col else None
            residue = flank[len(flank)//2].upper() if flank and len(flank) > 0 else "S"

            yield PTMObservation(
                source_db="dbPTM",
                uniprot_id=u_id,
                position=pos,
                residue=residue,
                canonical_ptm_type=canonicalize_ptm_type(ptm_raw),
                raw_ptm_name=ptm_raw,
                evidence_tier="experimental",
            )


# Keyword lookup table for UniProt description-to-residue mapping
_FEATURE_RESIDUE_KEYWORDS = (
    ("threonine", "T"),
    ("tyrosine",  "Y"),
    ("lysine",    "K"),
    ("asparagine", "N"),
    ("cysteine",  "C"),
)


@register_reader(
    name="uniprot_features",
    header_keywords=["feature key", "position(s)"],
    file_keywords=["uniprot", "features"],
)
def _parse_uniprot_features(path: Path, **kwargs: Any) -> Iterator[PTMObservation]:
    """Parses exported UniProt Feature tables (TSV)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
        if not header_line:
            return
        headers = [h.strip().lower() for h in header_line.split("\t")]

        entry_idx = headers.index("entry") if "entry" in headers else 0
        key_idx = headers.index("feature key") if "feature key" in headers else 1
        pos_idx = headers.index("position(s)") if "position(s)" in headers else 2
        desc_idx = headers.index("description") if "description" in headers else 3
        max_idx = max(entry_idx, key_idx, pos_idx)

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max_idx:
                continue

            feature_key = parts[key_idx].strip().upper()
            if feature_key not in ("MOD_RES", "CARBOHYD", "LIPID"):
                continue

            match = re.search(r"^(\d+)$", parts[pos_idx].strip())
            if not match:
                continue

            u_id = parts[entry_idx].strip()
            pos = int(match.group(1))
            desc = parts[desc_idx].strip() if len(parts) > desc_idx else feature_key
            desc_lower = desc.lower()

            # Table-driven residue resolution
            residue = next((res for kw, res in _FEATURE_RESIDUE_KEYWORDS if kw in desc_lower), "S")

            yield PTMObservation(
                source_db="UniProt",
                uniprot_id=u_id,
                position=pos,
                residue=residue,
                canonical_ptm_type=canonicalize_ptm_type(desc),
                raw_ptm_name=desc,
                evidence_tier="curated",
            )


# Standard Column Aliases for auto-detecting generic TSV/CSV tables
COLUMN_ALIASES: Dict[str, List[str]] = {
    "uniprot_id": ["acc_id", "uniprot", "uniprot_id", "entry", "protein_id", "accession"],
    "position":   ["pos", "position", "site", "residue_pos", "coord"],
    "residue":    ["res", "residue", "amino_acid", "aa"],
    "ptm_type":   ["ptm", "ptm_type", "modification", "mod_name", "mod_type"],
}


def _parse_generic_tabular(
    path: Path,
    column_mapping: Optional[Dict[str, str]] = None,
    delimiter: Optional[str] = None,
    source_name: str = "Generic",
    **kwargs: Any,
) -> Iterator[PTMObservation]:
    """Fallback reader that automatically resolves column names from synonyms or explicit mapping."""
    delim = delimiter or ("\t" if str(path).endswith((".tsv", ".tab", ".txt")) else ",")
    col_map = dict(column_mapping) if column_mapping else {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Find first non-comment header line
        header_line = next((line.strip() for line in f if line.strip() and not line.strip().startswith("#")), "")
        if not header_line:
            return

        headers = [h.strip().lower() for h in header_line.split(delim)]

        def find_col(field_name: str) -> Optional[int]:
            if field_name in col_map and col_map[field_name].lower() in headers:
                return headers.index(col_map[field_name].lower())
            for alias in COLUMN_ALIASES.get(field_name, [field_name]):
                if alias in headers:
                    return headers.index(alias)
            return None

        id_idx = find_col("uniprot_id")
        pos_idx = find_col("position")
        res_idx = find_col("residue")
        ptm_idx = find_col("ptm_type")

        if id_idx is None or pos_idx is None:
            raise ValueError(f"Could not locate required columns (uniprot_id, position) in headers: {headers}")

        max_idx = max(id_idx, pos_idx)

        for line in f:
            parts = line.strip().split(delim)
            if len(parts) <= max_idx:
                continue

            try:
                pos = int(parts[pos_idx].strip())
            except ValueError:
                continue

            u_id = parts[id_idx].strip()
            res = parts[res_idx].strip().upper() if res_idx is not None and len(parts) > res_idx else "S"
            ptm_raw = parts[ptm_idx].strip() if ptm_idx is not None and len(parts) > ptm_idx else "Phosphorylation"

            yield PTMObservation(
                source_db=source_name,
                uniprot_id=u_id,
                position=pos,
                residue=res,
                canonical_ptm_type=canonicalize_ptm_type(ptm_raw),
                raw_ptm_name=ptm_raw,
                evidence_tier="curated",
            )


def read_ptm_sites(
    source: Union[str, Path],
    format: str = "auto",
    **kwargs: Any,
) -> Iterator[PTMObservation]:
    """
    Deep unified reader for any PTM file.
    Auto-detects format from headers or filename, with zero caller-side boilerplate.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"PTM data source not found: {path}")

    # 1. Explicit format routing
    fmt_key = format.lower().strip()
    if fmt_key in _READER_REGISTRY:
        return _READER_REGISTRY[fmt_key]["func"](path, **kwargs)

    # 2. Header and filename signature matching
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        snippet = " ".join(f.readline().lower() for _ in range(5))

    file_tag = str(path).lower()

    for reader in _READER_REGISTRY.values():
        headers_match = reader["header_keywords"] and all(k in snippet for k in reader["header_keywords"])
        filename_match = reader["file_keywords"] and any(k in file_tag for k in reader["file_keywords"])

        if headers_match or filename_match:
            return reader["func"](path, **kwargs)

    # 3. Fallback to smart generic tabular reader
    return _parse_generic_tabular(path, **kwargs)
