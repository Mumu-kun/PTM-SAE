"""Fetcher, sequence validator, and multi-label site harmonization engine."""

import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from ptm_sae.data.ingestion import get_stratum_for_residue
from ptm_sae.data.schema import Protein, PTMObservation, UnifiedResidueSite

_DEFAULT_USER_AGENT = "Modular-PTM-SAE-Engine/1.0 (Academic Research; BUET)"


def compute_sha256(filepath: str | Path) -> str:
    """Compute SHA256 checksum of a file for provenance verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _open_http_stream(url: str, timeout: int = 120):
    """Open an HTTP stream with standardized research headers and timeout."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _DEFAULT_USER_AGENT},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _stream_download_to_file(
    url: str, destination_path: Path, chunk_size: int = 65536
) -> None:
    """Stream download a remote URL directly to a local file atomically via .part file."""
    temp_path = destination_path.with_suffix(destination_path.suffix + ".part")
    with _open_http_stream(url) as response, open(temp_path, "wb") as out_file:
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
    temp_path.replace(destination_path)


def _record_manifest(
    entry_key: str, source_name: str, url_or_path: str, file_path: Path
) -> None:
    """Record source provenance metadata in data/acquisition_manifest.json."""
    sha = compute_sha256(file_path)
    size = file_path.stat().st_size

    manifest_path = Path("data/acquisition_manifest.json")
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {}

    manifest_data[entry_key] = {
        "source_name": source_name,
        "url_or_path": url_or_path,
        "sha256": sha,
        "file_size_bytes": size,
        "timestamp_utc": str(file_path.stat().st_mtime),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def fetch_uniprot_human_proteome(
    output_dir: str | Path = "data/raw/uniprot",
    reviewed: bool = True,
    out_dir: str | Path | None = None,
    taxonomy_id: int = 9606,
    force_download: bool = False,
) -> Path:
    """
    Stream and cache the reviewed human proteome FASTA from UniProt REST API.
    Records provenance and SHA256 in data/acquisition_manifest.json.
    """
    resolved_dir = Path(out_dir if out_dir is not None else output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = resolved_dir / "human_reviewed_canonical.fasta"

    # Reuse existing cache if file exists and has content
    if fasta_path.exists() and not force_download and fasta_path.stat().st_size > 0:
        return fasta_path

    # Stream download directly from UniProt REST endpoint
    query = (
        f"(reviewed:{'true' if reviewed else 'false'}) AND (taxonomy_id:{taxonomy_id})"
    )
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query={urllib.parse.quote(query)}"

    _stream_download_to_file(url, fasta_path)

    _record_manifest(
        entry_key="uniprot_human_proteome",
        source_name="UniProt Swiss-Prot Human Proteome",
        url_or_path=url,
        file_path=fasta_path,
    )
    return fasta_path


def fetch_cplm_human(
    output_dir: str | Path = "data/raw",
    force_download: bool = False,
) -> Path:
    """
    Stream and cache human lysine PTM events from CPLM 4.0 (Homo sapiens.zip).
    Records cryptographic provenance and SHA256 in data/acquisition_manifest.json.
    """
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    cplm_path = resolved_dir / "cplm_human.txt"

    if cplm_path.exists() and not force_download and cplm_path.stat().st_size > 0:
        return cplm_path

    url = "http://cplm.biocuckoo.cn/Download/Homo%20sapiens.zip"
    with _open_http_stream(url) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        target_name = next(
            (
                name
                for name in archive.namelist()
                if "homo" in name.lower() and name.endswith(".txt")
            ),
            archive.namelist()[0],
        )
        temp_path = cplm_path.with_suffix(".part")
        with archive.open(target_name) as in_file, open(temp_path, "wb") as out_file:
            while chunk := in_file.read(65536):
                out_file.write(chunk)
        temp_path.replace(cplm_path)

    _record_manifest(
        entry_key="cplm_human_proteome",
        source_name="CPLM 4.0 Human Lysine Modifications",
        url_or_path=url,
        file_path=cplm_path,
    )
    return cplm_path


def fetch_uniprot_ptm_features(
    output_dir: str | Path = "data/raw",
    reviewed: bool = True,
    taxonomy_id: int = 9606,
    force_download: bool = False,
) -> Path:
    """
    Stream reviewed human PTM feature tables from UniProt REST API (MOD_RES, CARBOHYD)
    and parse into standardized canonical TSV (Entry, Feature key, Position(s), Description).
    Records provenance and SHA256 in data/acquisition_manifest.json.
    """
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = resolved_dir / "uniprot_features.tsv"

    if tsv_path.exists() and not force_download and tsv_path.stat().st_size > 0:
        return tsv_path

    query = (
        f"(reviewed:{'true' if reviewed else 'false'}) AND (taxonomy_id:{taxonomy_id})"
    )
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=tsv&query={urllib.parse.quote(query)}&fields=accession,ft_mod_res,ft_carbohyd"

    feature_regex = re.compile(r'(MOD_RES|CARBOHYD)\s+(\d+)(?:;\s*/note="([^"]+)")?')
    temp_path = tsv_path.with_suffix(".part")

    with (
        _open_http_stream(url) as response,
        open(temp_path, "w", encoding="utf-8") as out_file,
    ):
        out_file.write("Entry\tFeature key\tPosition(s)\tDescription\n")
        response.readline()  # Skip TSV header
        while raw_line := response.readline():
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split("	")
            if len(parts) < 2:
                continue

            accession = parts[0].strip()
            feature_text = " ; ".join(parts[1:])
            for match in feature_regex.finditer(feature_text):
                feature_key = match.group(1)
                position = match.group(2)
                description = match.group(3) or feature_key
                out_file.write(
                    f"{accession}\t{feature_key}\t{position}\t{description}\n"
                )

    temp_path.replace(tsv_path)

    _record_manifest(
        entry_key="uniprot_human_ptm_features",
        source_name="UniProt Swiss-Prot Human PTM Features (ft_mod_res, ft_carbohyd)",
        url_or_path=url,
        file_path=tsv_path,
    )
    return tsv_path


def harmonize_and_validate_ptm_sites(
    observations: Iterable[PTMObservation],
    sequences: dict[str, Protein],
) -> tuple[list[UnifiedResidueSite], list[dict]]:
    """
    Validates biological invariants across all PTM observations:
    1. Sequence existence: uniprot_id exists in canonical sequences.
    2. Coordinate bounds: 1 <= position <= len(sequence).
    3. Residue match: sequence[position - 1] == observation.residue.

    Mismatches are routed to audit_log.
    Valid observations are harmonized into UnifiedResidueSite models.
    """
    audit_log: list[dict] = []
    grouped: dict[tuple[str, int], list[PTMObservation]] = defaultdict(list)

    def _record_mismatch(obs: PTMObservation, actual: str | None, reason: str):
        audit_log.append(
            {
                "uniprot_id": obs.uniprot_id,
                "position": obs.position,
                "expected_residue": obs.residue,
                "actual_residue": actual,
                "reason": reason,
                "source_db": obs.source_db,
                "ptm_type": obs.canonical_ptm_type,
            }
        )

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

    unified_sites: list[UnifiedResidueSite] = []

    # Harmonize grouped observations at identical coordinates into multi-label sites
    for (uniprot_id, position), observation_group in grouped.items():
        representative_obs = observation_group[0]
        residue = representative_obs.residue.upper()
        stratum = get_stratum_for_residue(residue)

        ptm_types = set()
        sources = set()
        multiplicity: dict[str, int] = defaultdict(int)
        evidence_tiers: dict[str, str] = {}

        for obs in observation_group:
            canonical_type = obs.canonical_ptm_type
            ptm_types.add(canonical_type)
            sources.add(obs.source_db)
            multiplicity[canonical_type] += 1
            if obs.evidence_tier:
                evidence_tiers[canonical_type] = obs.evidence_tier

        unified = UnifiedResidueSite(
            uniprot_id=uniprot_id,
            position=position,
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
