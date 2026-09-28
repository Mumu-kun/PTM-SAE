"""M1 -- Acquisition. Ported from N1.ipynb cell 5.

Inputs: none. Emits: raw data under `CorpusPaths.raw_dir`; a Manifest (SHA sums + access dates).
Sources: UniProt/Swiss-Prot, CPLM 4.0 (human), dbPTM 2025 (37 types), qPTM (Human/Mouse/Rat/
Yeast), O-GlcNAcAtlas 5.0, N-GlycositeAtlas, N-GlyDE (via StackGlyEmbed).

Kaggle-path assumptions removed -- every function here takes `raw_dir: Path` explicitly (callers
pass `CorpusPaths.raw_dir`). OOD-species acquisition (mouse/rat/yeast/E. coli) is N1's own M2/M3
extension and is not ported here; this subpackage's scope is the human corpus only.

MOCK_MODE (the `mock` kwarg on every `acquire_*`) writes small synthetic fixtures shaped like each
real file's documented schema, exercising every parser with no network access -- this is what
`corpus/pipeline.py`'s local smoke test runs against.
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ptm_sae.corpus.config import CorpusPaths, Manifest

DBPTM_TYPE_URLS: dict[str, str] = {
    "N-linked Glycosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/N-linked Glycosylation.gz",
    "O-linked Glycosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/O-linked Glycosylation.gz",
    "Phosphorylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Phosphorylation.gz",
    "Deamidation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Deamidation.gz",
    "S-nitrosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/S-nitrosylation.gz",
    "S-palmitoylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/S-palmitoylation.gz",
    "Glutathionylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Glutathionylation.gz",
    "Sulfhydration": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Sulfhydration.gz",
    "Methylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Methylation.gz",
    "Citrullination": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Citrullination.gz",
    "ADP-ribosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/ADP-ribosylation.gz",
    "Nitration": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Nitration.gz",
    "Sulfation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Sulfation.gz",
    "Acetylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Acetylation.gz",
    "Biotinylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Biotinylation.gz",
    "Butyrylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Butyrylation.gz",
    "Carboxyethylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Carboxyethylation.gz",
    "Carboxylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Carboxylation.gz",
    "Crotonylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Crotonylation.gz",
    "Formylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Formylation.gz",
    "Glutarylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Glutarylation.gz",
    "Lactylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Lactylation.gz",
    "Lipoylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Lipoylation.gz",
    "Malonylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Malonylation.gz",
    "Neddylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Neddylation.gz",
    "Propionylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Propionylation.gz",
    "Sumoylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Sumoylation.gz",
    "Succinylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Succinylation.gz",
    "Ubiquitination": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Ubiquitination.gz",
    "Hydroxylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Hydroxylation.gz",
    "S-cysteinylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/S-cysteinylation.gz",
    "S-linked Glycosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/S-linked Glycosylation.gz",
    "Iodination": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Iodination.gz",
    "Gamma-carboxyglutamic acid": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Gamma-carboxyglutamic acid.gz",
    "Sulfoxidation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Sulfoxidation.gz",
    "C-linked Glycosylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/C-linked Glycosylation.gz",
    "Serotonylation": "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Serotonylation.gz",
}

CPLM_BASE = "https://cplm.biocuckoo.cn/Download/"
CPLM_SPECIES_HUMAN = "Homo sapiens"

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_FEATURE_FIELDS = (
    "accession,length,sequence,ft_carbohyd,ft_mod_res,ft_lipid,ft_signal,ft_transit"
)

OGLCNAC_URLS: dict[str, str] = {
    "Dataset-I": "https://oglcnac.org/static/dataset/Atlas%205.0_unambiguous%20sites_20251208.csv?v=20260523",
    "Dataset-II": "https://oglcnac.org/static/dataset/Atlas%205.0_ambiguous%20sites_20251208.csv?v=20260523",
}

STACKGLYEMBED_BASE = (
    "https://raw.githubusercontent.com/nafcoder/StackGlyEmbed/main/Dataset/"
)
GLYCOSITE_ATLAS_URL = "http://nglycositeatlas.biomarkercenter.org/download/HumanAll/"

QPTM_FORM_URL = "https://qptm.omicsbio.info/download.php"
QPTM_DIRECT_URL: str | None = os.environ.get("QPTM_ACCESS_TOKEN")
QPTM_NOTE = (
    "QPTM_ACCESS_TOKEN is unset -- qptm.omicsbio.info/download.php is a gated request form, not "
    "a static file endpoint. Submit that form once by hand to get a real link, then set it as the "
    "QPTM_ACCESS_TOKEN environment variable (not in this file). "
    "acquire_qptm() raises rather than silently downloading the form page."
)


def download_file(
    url: str,
    dest: Path,
    timeout: int = 120,
    chunk_size: int = 1 << 20,
    max_retries: int = 4,
    backoff_base: float = 5.0,
) -> Path:
    """Streaming download with parent-dir creation, retry-with-backoff, and an ATOMIC write.
    Retries transient network errors (ConnectionError, Timeout, ChunkedEncodingError) up to
    `max_retries` times with exponential backoff; HTTPError (4xx/5xx) is NOT retried. Writes to a
    `.part` sibling and renames to `dest` only after a fully successful download, so a crash
    mid-download never leaves a truncated file sitting at the real destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            tmp_dest.rename(dest)
            return dest
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            last_exc = e
            tmp_dest.unlink(missing_ok=True)
            if attempt == max_retries:
                break
            wait = backoff_base * (2 ** (attempt - 1))
            print(
                f"[download_file] attempt {attempt}/{max_retries} failed ({type(e).__name__}: "
                f"{e}) for {url} -- retrying in {wait:.0f}s."
            )
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            tmp_dest.unlink(missing_ok=True)
            raise  # non-2xx is a real, non-transient failure -- never retry this one
    tmp_dest.unlink(missing_ok=True)
    raise RuntimeError(
        f"[download_file] Giving up after {max_retries} attempts for {url} -- "
        f"last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _download_and_extract_zip_member(
    url: str, dest: Path, timeout: int = 600, chunk_size: int = 1 << 20
) -> Path:
    """Downloads a ZIP, extracts its largest member to `dest` as raw bytes (decode deferred to
    whatever reads `dest` later). The ZIP download itself goes through `download_file` for
    retry-with-backoff + atomic write; only local zip extraction keeps its own `.ziptmp`
    handling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.with_suffix(dest.suffix + ".ziptmp")
    download_file(url, zip_path, timeout=timeout, chunk_size=chunk_size)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members:
            raise ValueError(
                f"_download_and_extract_zip_member: {url} zip has no file members"
            )
        member = max(members, key=lambda m: m.file_size)
        with zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, length=chunk_size)
    zip_path.unlink(missing_ok=True)
    return dest


def _resolve_raw(raw_dir: Path, rel_path: str) -> Path | None:
    candidate = raw_dir / rel_path
    return candidate if candidate.exists() else None


def acquire_uniprot(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> Path:
    """Human reviewed Swiss-Prot, length<=1022, with PTM feature fields."""
    dest_name = "swissprot_human_reviewed.tsv"
    dest = raw_dir / dest_name
    query = "(reviewed:true) AND (organism_id:9606) AND (length:[1 TO 1022])"
    url = (
        f"{UNIPROT_STREAM}?query={requests.utils.quote(query)}"
        f"&fields={requests.utils.quote(UNIPROT_FEATURE_FIELDS)}&format=tsv"
    )
    cached = None if force else _resolve_raw(raw_dir, dest_name)
    if cached is not None:
        manifest.log(
            "UniProt/Swiss-Prot (human)",
            url,
            cached,
            "CACHED (reused, not re-downloaded)",
        )
        return cached
    if mock:
        _write_mock_swissprot(dest)
    else:
        download_file(url, dest)
    manifest.log(
        "UniProt/Swiss-Prot (human)",
        url,
        dest,
        "Reviewed human, length<=1022, TSV with ft_carbohyd/ft_mod_res/ft_lipid/ft_signal/ft_transit",
    )
    return dest


def acquire_cplm(
    raw_dir: Path,
    manifest: Manifest,
    species: list[str],
    mock: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Downloads CPLM per-species zips (Section 1 -- Lysine Modification Data -- only)."""
    out = {}
    for sp in species:
        rel = f"cplm/{sp.replace('/', '_')}.zip"
        dest = raw_dir / rel
        url = CPLM_BASE + requests.utils.quote(f"{sp}.zip")
        cached = None if force else _resolve_raw(raw_dir, rel)
        if cached is not None:
            manifest.log(
                f"CPLM 4.0 [{sp}]", url, cached, "CACHED (reused, not re-downloaded)"
            )
            out[sp] = cached
            continue
        if mock:
            _write_mock_cplm_zip(dest, n_sites=200)
        else:
            download_file(url, dest)
        manifest.log(
            f"CPLM 4.0 [{sp}]", url, dest, "Section 1 (Lysine Modification Data) only"
        )
        out[sp] = dest
    return out


def acquire_qptm(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> Path:
    """CF-26: the independent-curation replication label set -- Human/Mouse/Rat/Yeast site-level
    PTMs in one table (a ZIP containing one large TSV, extracted via
    `_download_and_extract_zip_member`)."""
    rel = "qptm/qptm_all_data.tsv"
    dest = raw_dir / rel
    cached = None if force else _resolve_raw(raw_dir, rel)
    if cached is not None:
        manifest.log(
            "qPTM (CF-26 replication set)",
            QPTM_DIRECT_URL or QPTM_FORM_URL,
            cached,
            "CACHED (reused, not re-downloaded)",
        )
        return cached
    if mock:
        _write_mock_qptm(dest, n_sites=180)
    else:
        if not QPTM_DIRECT_URL:
            raise RuntimeError(QPTM_NOTE)
        _download_and_extract_zip_member(QPTM_DIRECT_URL, dest)
    manifest.log(
        "qPTM (CF-26 replication set)",
        QPTM_DIRECT_URL or QPTM_FORM_URL,
        dest,
        "Human/Mouse/Rat/Yeast; species filter to human applied downstream (M3)",
    )
    return dest


def parse_qptm_flatfile(path: Path) -> pd.DataFrame:
    """qPTM site table -> (accession, position, type, species, pmid, reliability)."""
    raw = _read_flatfile_tsv(path)
    col_map = {
        "accession": _match_column(
            list(raw.columns), ["uniprot", "accession", "protein", "protein id"]
        ),
        "position": _match_column(list(raw.columns), ["position", "site"]),
        "type": _match_column(
            list(raw.columns),
            ["modification", "ptm type", "modification type", "ptm", "type"],
        ),
        "species": _match_column(list(raw.columns), ["species", "organism"]),
        "pmid": _match_column(list(raw.columns), ["pmid", "reference"]),
        "reliability": _match_column(list(raw.columns), ["reliability"]),
    }
    missing = [
        k for k, v in col_map.items() if v is None and k not in ("pmid", "reliability")
    ]
    if missing:
        raise ValueError(
            f"parse_qptm_flatfile: missing columns {missing} in {path.name}: {list(raw.columns)}"
        )
    out = pd.DataFrame(
        {
            "accession": raw[col_map["accession"]].astype(str),
            "position": pd.to_numeric(raw[col_map["position"]], errors="coerce").astype(
                "Int64"
            ),
            "type": raw[col_map["type"]].astype(str),
            "species": raw[col_map["species"]].astype(str),
            "pmid": raw[col_map["pmid"]].astype(str) if col_map["pmid"] else pd.NA,
            "reliability": (
                pd.to_numeric(raw[col_map["reliability"]], errors="coerce").astype(
                    "Int64"
                )
                if col_map["reliability"]
                else pd.NA
            ),
        }
    )
    return out.dropna(subset=["accession", "position"])


def acquire_dbptm(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> dict[str, Path]:
    if not mock and not DBPTM_TYPE_URLS:
        raise RuntimeError(
            "DBPTM_TYPE_URLS is empty -- see N1_CHANGELOG.md for the confirmed endpoint list."
        )
    out = {}
    urls = (
        DBPTM_TYPE_URLS
        if (not mock or DBPTM_TYPE_URLS)
        else {
            "N-linked Glycosylation": "MOCK",
            "Succinylation": "MOCK",
            "O-linked Glycosylation": "MOCK",
        }
    )
    for ptm_type, url in urls.items():
        rel = f"dbptm/{ptm_type.replace(' ', '_')}.txt.gz"
        dest = raw_dir / rel
        cached = None if force else _resolve_raw(raw_dir, rel)
        if cached is not None:
            manifest.log(
                f"dbPTM 2025 [{ptm_type}]",
                url,
                cached,
                "CACHED (reused, not re-downloaded)",
            )
            out[ptm_type] = cached
            continue
        if mock:
            _write_mock_dbptm_flatfile(dest, ptm_type, n_sites=150)
        else:
            download_file(url, dest)
        manifest.log(
            f"dbPTM 2025 [{ptm_type}]", url, dest, "Experimentally validated only"
        )
        out[ptm_type] = dest
    return out


def acquire_oglcnac_atlas(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> dict[str, Path]:
    """Dataset-I = unambiguous sites (feeds positives); Dataset-II = ambiguous sites, routed to
    M3's ambiguity mask purely by the raw_sources dict key containing 'oglcnac_dataset_ii'."""
    out = {}
    for dataset in ["Dataset-I", "Dataset-II"]:
        rel = f"oglcnac_atlas/{dataset}.csv"
        dest = raw_dir / rel
        url = OGLCNAC_URLS[dataset]
        cached = None if force else _resolve_raw(raw_dir, rel)
        if cached is not None:
            manifest.log(
                f"O-GlcNAcAtlas 5.0 [{dataset}]",
                url,
                cached,
                "CACHED (reused, not re-downloaded)",
            )
            out[dataset] = cached
            continue
        if mock:
            _write_mock_oglcnac(dest, dataset, n_sites=100)
        else:
            download_file(url, dest)
        manifest.log(
            f"O-GlcNAcAtlas 5.0 [{dataset}]",
            url,
            dest,
            "Dataset-I=unambiguous, Dataset-II=ambiguous (drives the ambiguity mask, M3 step 3a)",
        )
        out[dataset] = dest
    return out


def acquire_nglycosite_atlas(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> Path:
    """Supplementary N-glycosylation evidence (unioned with dbPTM's own 'N-linked Glycosylation'
    entries), fetched directly from the database's own site as XLSX."""
    rel = "n_glycosite_atlas/HumanAll.xlsx"
    dest = raw_dir / rel
    cached = None if force else _resolve_raw(raw_dir, rel)
    if cached is not None:
        manifest.log(
            "N-GlycositeAtlas",
            GLYCOSITE_ATLAS_URL,
            cached,
            "CACHED (reused, not re-downloaded)",
        )
        return cached
    if mock:
        _write_mock_glycosite_atlas_xlsx(dest, n_sites=120)
    else:
        download_file(GLYCOSITE_ATLAS_URL, dest)
    manifest.log(
        "N-GlycositeAtlas",
        GLYCOSITE_ATLAS_URL,
        dest,
        "XLSX, confirmed schema: Database Accession (UniProtKB)/Glycosylation location/...",
    )
    return dest


def acquire_nglyde(
    raw_dir: Path, manifest: Manifest, mock: bool = False, force: bool = False
) -> dict[str, Path]:
    """The only Gold-tier negative source (via build_gold_negatives_nglyde). Sourced from
    StackGlyEmbed's GitHub reproduction of the N-GlyDE benchmark, using the '_original'
    identity-threshold variant (matching the 2019 paper's own benchmark)."""
    out = {}
    for split in ["train", "test"]:
        rel = f"n_glyde/fasta_{split}_NGlyDE_original.txt"
        dest = raw_dir / rel
        url = STACKGLYEMBED_BASE + f"fasta_{split}_NGlyDE_original.txt"
        cached = None if force else _resolve_raw(raw_dir, rel)
        if cached is not None:
            manifest.log(
                f"N-GlyDE [{split}, via StackGlyEmbed repo]",
                url,
                cached,
                "CACHED (reused, not re-downloaded)",
            )
            out[split] = cached
            continue
        if mock:
            _write_mock_stackglyembed_fasta_pairs(
                dest, n_proteins=15, seed=5 if split == "train" else 50
            )
        else:
            download_file(url, dest)
        manifest.log(
            f"N-GlyDE [{split}, via StackGlyEmbed repo]",
            url,
            dest,
            "Gold-tier positives+negatives -- see build_gold_negatives_nglyde",
        )
        out[split] = dest
    return out


# ---------------------------------------------------------------------------
# Parsers -- each source's raw file(s) -> a normalized (accession, position, type, ...) table.
# ---------------------------------------------------------------------------
def _match_column(columns: list[str], candidates: list[str]) -> str | None:
    cols_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        for lc, orig in cols_lower.items():
            if cand in lc:
                return orig
    return None


def _decode_bytes_robust(data: bytes) -> str:
    """UTF-8, falling back to cp1252 then latin-1 for files that aren't UTF-8 -- lab-exported
    CSVs commonly carry stray cp1252 bytes in free-text columns. latin-1 is the final fallback
    since it cannot raise UnicodeDecodeError."""
    for enc in ("utf-8", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _read_csv_robust(path_or_bytes, **kwargs) -> pd.DataFrame:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            data = f.read()
    return pd.read_csv(io.StringIO(_decode_bytes_robust(data)), **kwargs)


def _read_flatfile_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Reads a downloaded flat file as TSV regardless of what its name/extension claims. dbPTM's
    per-type '.gz' links are actually gzip-compressed TAR archives (PAX format), not bare gzip --
    content is sniffed (tarfile's signature check, then raw gzip magic bytes) rather than trusted
    from the filename."""
    try:
        with tarfile.open(path, mode="r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if not members:
                raise ValueError(
                    f"_read_flatfile_tsv: tar archive at {path.name} has no regular file members"
                )
            member = max(members, key=lambda m: m.size)
            with tf.extractfile(member) as f:
                data = f.read()
            return _read_csv_robust(data, sep="\t", **kwargs)
    except tarfile.ReadError:
        pass  # not a tar archive -- fall through to gzip/plain-text handling

    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            data = f.read()
        return _read_csv_robust(data, sep="\t", **kwargs)
    return _read_csv_robust(path, sep="\t", **kwargs)


CPLM_POSITIONAL_COLUMNS = [
    "cplm_id",
    "accession",
    "position",
    "type",
    "gene",
    "species",
    "sequence",
    "evidence_type",
    "pmid",
]


def parse_cplm_zip(zip_path: Path) -> pd.DataFrame:
    """CPLM's real per-species zip has no header row -- read positionally against
    CPLM_POSITIONAL_COLUMNS."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        data_member = max(members, key=lambda m: m.file_size)
        with zf.open(data_member) as f:
            raw = _read_csv_robust(
                f.read(), sep="\t", header=None, names=CPLM_POSITIONAL_COLUMNS
            )

    out = pd.DataFrame(
        {
            "accession": raw["accession"].astype(str),
            "position": pd.to_numeric(raw["position"], errors="coerce").astype("Int64"),
            "type": raw["type"].astype(str),
            "pmid": raw["pmid"].astype(str),
            "flanking_peptide": pd.NA,
        }
    )
    return out.dropna(subset=["accession", "position", "type"])


DBPTM_POSITIONAL_COLUMNS = [
    "entry_name",
    "accession",
    "position",
    "type",
    "pmid",
    "flanking_peptide",
]


def parse_dbptm_flatfile(path: Path, ptm_type: str) -> pd.DataFrame:
    """dbPTM per-type experimental flat file -> (accession, position, type, pmid, evidence).
    No header row -- read positionally. `evidence` is hard-coded "experimental" since every row
    is already dbPTM's own experimentally-validated tier (the site's dedicated
    /download/experiment/ endpoint)."""
    raw = _read_flatfile_tsv(path, header=None, names=DBPTM_POSITIONAL_COLUMNS)
    entry_name = raw["entry_name"].astype(str)
    out = pd.DataFrame(
        {
            "accession": raw["accession"].astype(str),
            "position": pd.to_numeric(raw["position"], errors="coerce").astype("Int64"),
            "type": ptm_type,
            "pmid": raw["pmid"].astype(str),
            "evidence": "experimental",
            "species_suffix": entry_name.str.rsplit("_", n=1).str[-1],
        }
    )
    return out.dropna(subset=["accession", "position"])


def parse_oglcnac_atlas_flatfile(path: Path, species: str = "human") -> pd.DataFrame:
    """O-GlcNAcAtlas 5.0 Dataset-I/II CSV -> (accession, position, type, evidence). Filtered to
    species=="human" AND accession_source=="UniProt" AND a non-empty position_in_protein (not
    position_in_peptide, which is within-peptide, not within-protein)."""
    raw = _read_csv_robust(path)  # comma-separated, not tab
    human = raw[
        (
            raw["species"].astype(str).str.strip().str.lower()
            == str(species).strip().lower()
        )
        & (raw["accession_source"].astype(str).str.strip().str.lower() == "uniprot")
    ].copy()
    human["position_in_protein"] = pd.to_numeric(
        human["position_in_protein"], errors="coerce"
    )
    human = human.dropna(subset=["accession", "position_in_protein"])
    out = pd.DataFrame(
        {
            "accession": human["accession"].astype(str),
            "position": human["position_in_protein"].astype(int),
            "type": "O-GlcNAcylation",
            "evidence": "MS/MS",
        }
    )
    return out.dropna(subset=["accession", "position"])


def parse_stackglyembed_fasta_pairs(path: Path) -> pd.DataFrame:
    """Two lines per protein: an odd header 'accession,label,position[,label,position,...]'
    (1=glycosylated, 0=validated non-glycosylated sequon), even line = raw sequence (unused --
    sequences come from the UniProt-derived corpus, matched by accession)."""
    lines = _decode_bytes_robust(path.read_bytes()).strip().split("\n")
    rows = []
    for i in range(0, len(lines) - 1, 2):
        header = lines[i].strip()
        parts = header.split(",")
        if not parts or not parts[0]:
            continue
        accession, pairs = parts[0], parts[1:]
        for j in range(0, len(pairs) - 1, 2):
            label_str, pos_str = pairs[j].strip(), pairs[j + 1].strip()
            if not pos_str.lstrip("-").isdigit():
                continue
            rows.append(
                {
                    "UniProt Accession": accession,
                    "Position": int(pos_str),
                    "Label": "glycosylated" if label_str == "1" else "non-glycosylated",
                }
            )
    return pd.DataFrame(rows, columns=["UniProt Accession", "Position", "Label"])


def _find_excel_header_row(
    raw_no_header: pd.DataFrame, must_contain: list[str], max_scan_rows: int = 10
) -> int:
    """Scans the first rows for the one containing all of `must_contain` (case-insensitive
    substrings), since journal supplementary XLSX exports commonly ship a title row before the
    real header row."""
    n_scan = min(max_scan_rows, len(raw_no_header))
    for row_idx in range(n_scan):
        row_text = " | ".join(
            str(v).strip().lower() for v in raw_no_header.iloc[row_idx].tolist()
        )
        if all(token in row_text for token in must_contain):
            return row_idx
    raise ValueError(
        f"_find_excel_header_row: no row in the first {n_scan} rows contains all of "
        f"{must_contain}; first {n_scan} rows were: {raw_no_header.head(n_scan).values.tolist()}"
    )


def parse_nglycosite_atlas_flatfile(path: Path) -> pd.DataFrame:
    """N-GlycositeAtlas's HumanAll.xlsx -> (accession, position, type, evidence). Real header row
    located dynamically (see _find_excel_header_row) since the sheet has a title row above the
    real column-name row."""
    raw_no_header = pd.read_excel(path, header=None)
    header_row = _find_excel_header_row(
        raw_no_header, must_contain=["uniprotkb", "glycosylation location"]
    )
    raw = pd.read_excel(path, header=header_row)
    col_map = {
        "accession": _match_column(
            list(raw.columns), ["uniprotkb", "accession", "database accession"]
        ),
        "position": _match_column(
            list(raw.columns), ["glycosylation location", "position", "location"]
        ),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(
            f"parse_nglycosite_atlas_flatfile: missing columns {missing} in {path.name}: {list(raw.columns)}"
        )
    out = pd.DataFrame(
        {
            "accession": raw[col_map["accession"]].astype(str),
            "position": pd.to_numeric(raw[col_map["position"]], errors="coerce").astype(
                "Int64"
            ),
            "type": "N-glycosylation",
            "evidence": "MS/MS",
        }
    )
    return out.dropna(subset=["accession", "position"])


def parse_nglyde_flatfile(path: Path) -> pd.DataFrame:
    """N-GlyDE (via StackGlyEmbed) -> (UniProt Accession, Position, Label). Feeds the Gold-tier
    NEGATIVE set only (Label containing "non"); positives carried through too since M3 filters."""
    return parse_stackglyembed_fasta_pairs(path)


# ---------------------------------------------------------------------------
# Mock fixtures -- exercise every parser above with no network access. Shapes mirror the
# documented real schemas.
# ---------------------------------------------------------------------------
_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _random_seq(rng, length):
    return "".join(rng.choice(list(_AA20)) for _ in range(length))


def _write_mock_swissprot(dest: Path, n_proteins: int = 200, seed: int = 1):
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    rows = []
    for i in range(n_proteins):
        length = rng.randint(50, 800)
        rows.append(
            {
                "Entry": f"P{10000 + i}",
                "Length": length,
                "Sequence": _random_seq(rng, length),
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, sep="\t", index=False)


def _write_mock_cplm_zip(dest: Path, n_sites: int = 200, seed: int = 2):
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    types = [
        "Succinylation",
        "Acetylation",
        "Ubiquitination",
        "Hydroxylation",
        "β-Hydroxybutyrylation",
    ]
    lines = []
    for i in range(n_sites):
        lines.append(
            "\t".join(
                [
                    str(i),
                    f"P{10000 + rng.randint(0, 199)}",
                    str(rng.randint(1, 500)),
                    rng.choice(types),
                    f"GENE{i}",
                    "Homo sapiens",
                    _random_seq(rng, 13),
                    "MS/MS",
                    str(20000000 + i),
                ]
            )
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr("Homo_sapiens.txt", "\n".join(lines))


def _write_mock_dbptm_flatfile(
    dest: Path, ptm_type: str, n_sites: int = 150, seed: int = 3
):
    import random

    rng = random.Random(seed + hash(ptm_type) % 1000)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    lines = []
    for i in range(n_sites):
        lines.append(
            "\t".join(
                [
                    f"MOCK{i}_HUMAN",
                    f"P{20000 + rng.randint(0, 149)}",
                    str(rng.randint(1, 500)),
                    ptm_type,
                    str(30000000 + i),
                    _random_seq(rng, 13),
                ]
            )
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))


def _write_mock_oglcnac(dest: Path, dataset: str, n_sites: int = 100, seed: int = 4):
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    rows = []
    for i in range(n_sites):
        comment = "authors’ note" if i == 3 else ""  # noqa: RUF001 -- deliberate non-ASCII char, see docstring above
        rows.append(
            {
                "id": i,
                "species": "human",
                "sample_type": "HeLa",
                "accession": f"P{30000 + (i % 40)}",
                "accession_source": "UniProt",
                "entry_name": f"MOCK{i}_HUMAN",
                "protein_name": "Mock protein",
                "gene_name": f"MOCK{i}",
                "peptide_seq": _random_seq(rng, 15),
                "site_residue": "S",
                "position_in_peptide": rng.randint(1, 15),
                "position_in_protein": rng.randint(1, 500),
                "method": "MS",
                "analytical_throughput": "HTP",
                "pmid": 20000000 + i,
                "condition": "mock/control",
                "log2Ratio": round(rng.uniform(-2, 2), 3),
                "p_value": "",
                "comments": comment,
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Written as cp1252, not UTF-8, to reproduce a real lab-exported CSV's stray Windows encoding.
    dest.write_bytes(pd.DataFrame(rows).to_csv(index=False).encode("cp1252"))


def _write_mock_qptm(dest: Path, n_sites: int = 180, seed: int = 6):
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    organisms = (
        ["Human"] * int(n_sites * 0.8)
        + ["Mouse"] * int(n_sites * 0.15)
        + ["Yeast"] * int(n_sites * 0.05)
    )
    while len(organisms) < n_sites:
        organisms.append("Human")
    rng.shuffle(organisms)
    ptm_types = ["Phosphorylation", "Acetylation", "Ubiquitination", "Succinylation"]
    rows = []
    for i, org in enumerate(organisms):
        rows.append(
            {
                "Organism": org,
                "PMID": 20000000 + i,
                "UniProt accession": f"P{40000 + (i % 60)}",
                "Gene name": f"MOCK{i % 60}",
                "Position": rng.randint(1, 500),
                "PTM": rng.choice(ptm_types),
                "Sequence window": _random_seq(rng, 15),
                "Raw peptide": _random_seq(rng, 12),
                "Sample": "MockCellLine",
                "Condition": "Treatment/Ctr",
                "Log2Ratio (peptide)": round(rng.uniform(-3, 3), 3),
                "P value (peptide)": round(rng.random(), 4),
                "Log2Ratio (protein)": "-",
                "P value (protein)": "-",
                "Reliability": rng.choice([2, 5]),
                "fdr (peptide)": round(rng.random() * 0.01, 6),
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, sep="\t", index=False)


def _write_mock_stackglyembed_fasta_pairs(
    dest: Path, n_proteins: int = 15, seed: int = 4
):
    """Positions land on real 'N' residues so the mock data is chemistry-valid like the real
    data (every glycosylation site sits on an asparagine)."""
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    lines = []
    for i in range(n_proteins):
        length = rng.randint(60, 200)
        seq = list(_random_seq(rng, length))
        n_positions = rng.sample(range(1, length + 1), k=min(4, length))
        for p in n_positions:
            seq[p - 1] = "N"
        seq_str = "".join(seq)
        pairs = []
        for p in n_positions:
            label = 1 if rng.random() > 0.4 else 0
            pairs.extend([str(label), str(p)])
        lines.append(f"P{70000 + i}," + ",".join(pairs))
        lines.append(seq_str)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")


def _write_mock_glycosite_atlas_xlsx(dest: Path, n_sites: int = 120, seed: int = 4):
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic mock-fixture generator, not crypto
    rows = []
    for i in range(n_sites):
        peptide = list(_random_seq(rng, 14))
        site_idx = rng.randrange(len(peptide))
        peptide[site_idx] = (
            "n"  # lowercase marks the modified residue, matching the real format
        )
        rows.append(
            {
                "Database Accession (UniProtKB)": f"P{50000 + (i % 40)}",
                "Gene names": f"MOCK{i % 40}",
                "Protein names": "Mock protein",
                "Glycosylation location": rng.randint(1, 500),
                "Glycosite-containing peptides": "".join(peptide),
                "Glycosite_20AAs": _random_seq(rng, 20),
                "Identification resources": rng.choice(["Liver", "Plasma", "Kidney"]),
                "Years": rng.randint(2010, 2022),
                "References": rng.randint(1, 100),
                "Reviewed": "",
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(dest, index=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_m1(
    paths: CorpusPaths, mock: bool = False, force: bool = False
) -> tuple[Manifest, dict[str, object]]:
    """Returns (manifest, resolved_paths) -- resolved_paths is the source of truth for where each
    artefact actually ended up. Downstream code MUST read paths from this dict."""
    from tqdm.auto import tqdm

    manifest = Manifest()
    raw_dir = paths.raw_dir
    print(f"[M1] mock={mock} force={force} raw_dir={raw_dir}")

    pbar = tqdm(total=7, desc="[M1] Acquisition", unit="source")
    resolved: dict[str, object] = {}
    pbar.set_postfix_str("swissprot")
    resolved["swissprot"] = acquire_uniprot(raw_dir, manifest, mock=mock, force=force)
    pbar.update(1)
    pbar.set_postfix_str("cplm")
    resolved["cplm"] = acquire_cplm(
        raw_dir, manifest, species=[CPLM_SPECIES_HUMAN], mock=mock, force=force
    )
    pbar.update(1)
    pbar.set_postfix_str("dbptm")
    resolved["dbptm"] = acquire_dbptm(raw_dir, manifest, mock=mock, force=force)
    pbar.update(1)
    pbar.set_postfix_str("qptm")
    try:
        resolved["qptm"] = acquire_qptm(
            raw_dir, manifest, mock=mock, force=force
        )  # CF-26: required, not optional
    except RuntimeError as e:
        if not mock:
            print(
                f"[M1] HELD: qPTM acquisition skipped ({e}). Re-run M1 once QPTM_ACCESS_TOKEN is set."
            )
        else:
            raise
    pbar.update(1)
    pbar.set_postfix_str("oglcnac_atlas")
    resolved["oglcnac_atlas"] = acquire_oglcnac_atlas(
        raw_dir, manifest, mock=mock, force=force
    )
    pbar.update(1)
    pbar.set_postfix_str("glycosite_atlas")
    resolved["glycosite_atlas"] = acquire_nglycosite_atlas(
        raw_dir, manifest, mock=mock, force=force
    )
    pbar.update(1)
    pbar.set_postfix_str("glyde")
    resolved["glyde"] = acquire_nglyde(raw_dir, manifest, mock=mock, force=force)
    pbar.update(1)
    pbar.close()

    manifest.save(paths.manifest)
    print(
        f"[M1] {len(manifest.entries)} artefacts logged -> {paths.manifest} "
        f"({sum(1 for e in manifest.entries if 'CACHED' in e['notes'])} reused from cache)"
    )
    return manifest, resolved
