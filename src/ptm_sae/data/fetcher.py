"""Fetcher, sequence validator, and multi-label site harmonization engine."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import urllib.request

from ptm_sae.data.ingestion import get_stratum_for_residue, parse_uniprot_fasta
from ptm_sae.data.schema import Protein, PTMObservation, UnifiedResidueSite


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA256 checksum of a file for provenance verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_uniprot_human_proteome(
    output_dir: Union[str, Path] = "data/raw/uniprot",
    reviewed: bool = True,
    taxonomy_id: int = 9606,
    force_download: bool = False,
) -> Path:
    """
    Stream and cache the reviewed human proteome FASTA from UniProt REST API.
    Records provenance and SHA256 in data/acquisition_manifest.json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "human_reviewed_canonical.fasta"

    # Reuse existing cache if file exists and has content
    if fasta_path.exists() and not force_download and fasta_path.stat().st_size > 0:
        return fasta_path

    # Stream download directly from UniProt REST endpoint
    query = f"(reviewed:{'true' if reviewed else 'false'}) AND (taxonomy_id:{taxonomy_id})"
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query={urllib.parse.quote(query)}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Modular-PTM-SAE-Engine/1.0 (Academic Research; BUET)"},
    )
    with urllib.request.urlopen(req, timeout=120) as response, open(fasta_path, "wb") as out_f:
        while chunk := response.read(65536):
            out_f.write(chunk)

    # Compute provenance metadata and record to manifest
    sha = compute_sha256(fasta_path)
    size = fasta_path.stat().st_size

    manifest_path = Path("data/acquisition_manifest.json")
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    manifest_data["uniprot_human_proteome"] = {
        "source_name": "UniProt Swiss-Prot Human Proteome",
        "url_or_path": url,
        "sha256": sha,
        "file_size_bytes": size,
        "timestamp_utc": str(Path(fasta_path).stat().st_mtime),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return fasta_path


def harmonize_and_validate_ptm_sites(
    observations: Iterable[PTMObservation],
    sequences: Dict[str, Protein],
) -> Tuple[List[UnifiedResidueSite], List[Dict]]:
    """
    Validates biological invariants across all PTM observations:
    1. Sequence existence: uniprot_id exists in canonical sequences.
    2. Coordinate bounds: 1 <= position <= len(sequence).
    3. Residue match: sequence[position - 1] == observation.residue.

    Mismatches are routed to audit_log.
    Valid observations are harmonized into UnifiedResidueSite models.
    """
    audit_log: List[Dict] = []
    grouped: Dict[Tuple[str, int], List[PTMObservation]] = defaultdict(list)

    def _record_mismatch(obs: PTMObservation, actual: Optional[str], reason: str):
        audit_log.append({
            "uniprot_id": obs.uniprot_id,
            "position": obs.position,
            "expected_residue": obs.residue,
            "actual_residue": actual,
            "reason": reason,
            "source_db": obs.source_db,
            "ptm_type": obs.canonical_ptm_type,
        })

    for obs in observations:
        # 1. Guard: Ensure protein exists in canonical sequences
        protein = sequences.get(obs.uniprot_id)
        if protein is None:
            _record_mismatch(obs, None, "protein_not_in_canonical_sequences")
            continue

        seq = protein.sequence
        pos = obs.position

        # 2. Guard: Ensure biological coordinate is within sequence bounds (1 <= pos <= L)
        if pos < 1 or pos > len(seq):
            _record_mismatch(obs, None, "coordinate_out_of_bounds")
            continue

        actual_res = seq[pos - 1].upper()
        expected_res = obs.residue.upper()

        # 3. Guard: Ensure sequence residue matches annotated site (reject isoform drift)
        if actual_res != expected_res:
            _record_mismatch(obs, actual_res, "sequence_residue_mismatch_isoform_drift")
            continue

        # Valid site: accumulate under (uniprot_id, position) for crosstalk aggregation
        grouped[(obs.uniprot_id, pos)].append(obs)

    unified_sites: List[UnifiedResidueSite] = []

    # Harmonize grouped observations at identical coordinates into multi-label sites
    for (u_id, pos), obs_list in grouped.items():
        first = obs_list[0]
        residue = first.residue.upper()
        stratum = get_stratum_for_residue(residue)

        ptm_types = set()
        sources = set()
        multiplicity: Dict[str, int] = defaultdict(int)
        evidence_tiers: Dict[str, str] = {}

        for obs in obs_list:
            c_type = obs.canonical_ptm_type
            ptm_types.add(c_type)
            sources.add(obs.source_db)
            multiplicity[c_type] += 1
            if obs.evidence_tier:
                evidence_tiers[c_type] = obs.evidence_tier

        unified = UnifiedResidueSite(
            uniprot_id=u_id,
            position=pos,
            residue=residue,
            stratum=stratum,
            ptm_types=ptm_types,
            sources=sources,
            source_multiplicity=dict(multiplicity),
            is_multi_label=len(ptm_types) > 1,
            metadata={"evidence_tiers": evidence_tiers},
        )
        unified_sites.append(unified)

    return unified_sites, audit_log
