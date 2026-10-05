"""Raw human proteome sequence fetcher.

The PTM-source fetchers that used to live here (`fetch_cplm_human`, `fetch_uniprot_ptm_features`)
and `harmonize_and_validate_ptm_sites` are retired -- superseded by the corpus build's chemistry-validated cascade
(now archived on git branch `archive/corpus-build`). Only
`fetch_uniprot_human_proteome` (sequence acquisition, not PTM labels) remains, unaffected by that
refactor.
"""

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

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
