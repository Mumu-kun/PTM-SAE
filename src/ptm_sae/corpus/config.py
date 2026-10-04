"""Paths, Config, and the caching/fingerprint scaffold shared by acquisition/clustering/labels.

Ported from the partner's `src/ptm_eval/N1.ipynb` (cell 3, "M0 -- Configuration & Design
Invariants"). The Kaggle-path branching (`/kaggle/working`, `/kaggle/input`, `/kaggle/temp`) is
removed entirely -- `CorpusPaths` is env-var driven instead, with no Kaggle/local flag.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ptm_sae.runtime import resolve_data_root


@dataclass
class CorpusPaths:
    """Corpus build directories. Each is overridable via its own env var; otherwise all three
    default under `PTM_SAE_CORPUS_ROOT` (default: the shared data root, `runtime.resolve_data_root()`)."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.interim_dir = Path(self.interim_dir)
        self.processed_dir = Path(self.processed_dir)
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> CorpusPaths:
        root = Path(os.environ.get("PTM_SAE_CORPUS_ROOT") or resolve_data_root())
        raw_dir = Path(
            os.environ.get("PTM_SAE_CORPUS_RAW_DIR", str(root / "data" / "raw"))
        )
        interim_dir = Path(
            os.environ.get("PTM_SAE_CORPUS_CACHE_DIR", str(root / "cache"))
        )
        processed_dir = Path(
            os.environ.get(
                "PTM_SAE_CORPUS_PROCESSED_DIR", str(root / "data" / "processed")
            )
        )
        return cls(
            raw_dir=raw_dir, interim_dir=interim_dir, processed_dir=processed_dir
        )

    @property
    def manifest(self) -> Path:
        return self.interim_dir / "manifest.json"

    @property
    def corpus_parquet(self) -> Path:
        return self.processed_dir / "corpus.parquet"

    @property
    def stratum_counts(self) -> Path:
        return self.processed_dir / "stratum_counts.json"

    @property
    def split_manifest(self) -> Path:
        return self.processed_dir / "split_manifest.json"

    @property
    def labels_stratified(self) -> Path:
        return self.processed_dir / "labels_stratified.parquet"

    @property
    def exclusion_mask(self) -> Path:
        return self.processed_dir / "exclusion_mask.parquet"

    @property
    def gold_negatives_nglyco(self) -> Path:
        return self.processed_dir / "gold_negatives_nglyco.parquet"


# ---------------------------------------------------------------------------
# Config -- ported verbatim from N1's cell 3 (only the fields M1/M2/M3 read; N1 itself already
# trimmed the original pipeline's Config down to this set, see the notebook's own [N1-CL#2]).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Config:
    # Corpus
    max_protein_length: int = 1022
    organism_primary: str = "human"
    swissprot_reviewed_only: bool = True

    # Label filters
    min_sites_per_type: int = 200
    min_sites_headline: int = 1000
    min_source_count_high_confidence: int = 2

    # qPTM's 0-5 reliability score floor (sites seen in >=N independent datasets). See
    # N1_CHANGELOG.md #22/#24 for the sensitivity sweep behind this default.
    qptm_reliability_floor: int = 2

    # Out-of-distribution species -- carried over from N1 for config-fingerprint parity, though
    # nothing in this subpackage builds the OOD corpora/labels (that is N1's own M2/M3 extension,
    # out of scope for this port).
    ood_species: dict = dataclasses.field(
        default_factory=lambda: {
            "mouse": {
                "taxon_id": "10090",
                "uniprot_suffix": "MOUSE",
                "cplm_species": "Mus musculus",
                "qptm_label": "mouse",
                "oglcnac_label": "mouse",
            },
            "rat": {
                "taxon_id": "10116",
                "uniprot_suffix": "RAT",
                "cplm_species": "Rattus norvegicus",
                "qptm_label": "rat",
                "oglcnac_label": "rat",
            },
            "yeast": {
                "taxon_id": "559292",
                "uniprot_suffix": "YEAST",
                "cplm_species": "Saccharomyces cerevisiae (strain ATCC 204508 or S288c)",
                "qptm_label": "yeast",
                "oglcnac_label": "yeast",
            },
            "ecoli": {
                "taxon_id": "83333",
                "uniprot_suffix": "ECOLI",
                "cplm_species": "Escherichia coli (strain K12)",
                "qptm_label": None,
                "oglcnac_label": None,
            },
        }
    )
    ood_min_sites_per_type: int = 200

    # Redundancy -- 40% identity, per the user's decision (see the implementation plan).
    cdhit_identity: float = 0.40
    cdhit_cluster_before_split: bool = True

    # Discovery / held-out split (cluster-level, assigned in M2)
    holdout_fraction: float = 0.20
    holdout_seed: int = 42
    holdout_stratify_by: str = "ptm_type"
    holdout_min_sites_per_type: int = 200

    # Residue strata -> chemistry-valid target residues (M2/M3). Codes match N1 exactly: K, ST,
    # N, C, R, Y, E, M, W, Q -- do not translate these to full words (see ingestion.py's
    # STRATUM_MAPPING, renamed to match).
    stratum_residues: dict = dataclasses.field(
        default_factory=lambda: {
            "K": {"K"},
            "ST": {"S", "T"},
            "N": {"N"},
            "C": {"C"},
            "R": {"R"},
            "Y": {"Y"},
            "E": {"E"},
            "M": {"M"},
            "W": {"W"},
            "Q": {"Q"},
        }
    )
    # PTM type -> stratum. Only the primary three are load-bearing for headline claims; the rest
    # exist so M3's "other PTM" pooling and crosstalk bookkeeping are complete.
    ptm_type_to_stratum: dict = dataclasses.field(
        default_factory=lambda: {
            # Lysine (K) -- CPLM's 29 verified types
            "succinylation": "K",
            "acetylation": "K",
            "crotonylation": "K",
            "malonylation": "K",
            "2-hydroxyisobutyrylation": "K",
            "beta-hydroxybutyrylation": "K",
            "butyrylation": "K",
            "propionylation": "K",
            "glutarylation": "K",
            "lactylation": "K",
            "formylation": "K",
            "benzoylation": "K",
            "HMGylation": "K",
            "MGcylation": "K",
            "MGylation": "K",
            "ubiquitination": "K",
            "sumoylation": "K",
            "pupylation": "K",
            "neddylation": "K",
            "methylation_K": "K",
            "glycation": "K",
            "hydroxylation_K": "K",
            "phosphoglycerylation": "K",
            "carboxymethylation": "K",
            "lipoylation": "K",
            "carboxylation": "K",
            "dietylphosphorylation": "K",
            "biotinylation": "K",
            "carboxyethylation": "K",
            # Ser/Thr
            "phosphorylation_ST": "ST",
            "O-GlcNAcylation": "ST",
            "O-glycosylation": "ST",
            # Asparagine
            "N-glycosylation": "N",
            "deamidation": "N",
            # Cysteine
            "S-nitrosylation": "C",
            "palmitoylation": "C",
            "glutathionylation": "C",
            "S-sulfhydration": "C",
            # Arginine
            "methylation_R": "R",
            "citrullination": "R",
            "ADP-ribosylation": "R",
            # Tyrosine
            "phosphorylation_Y": "Y",
            "nitration": "Y",
            "sulfation": "Y",
            "S-cysteinylation": "C",
            "S-linked Glycosylation": "C",
            "Iodination": "Y",
            "gamma-carboxyglutamic-acid": "E",
            "sulfoxidation": "M",
            "C-linked Glycosylation": "W",
            "serotonylation": "Q",
        }
    )
    # The three primary/headline types this thesis leads with.
    primary_types: tuple = ("N-glycosylation", "O-GlcNAcylation", "succinylation")


CFG = Config()


# ---------------------------------------------------------------------------
# Type canonicalisation -- hoisted in N1 so M_E and M3 share one mapping; kept here (M_E itself
# is not ported) since M2's raw-type stratification join and M3's cascade both need it.
# ---------------------------------------------------------------------------
PTM_TYPE_ALIASES: dict[str, str] = {
    "n-glycosylation": "N-glycosylation",
    "n-linked-glycosylation": "N-glycosylation",
    "asparagine-linked-glycosylation": "N-glycosylation",
    "o-glcnacylation": "O-GlcNAcylation",
    "o-glcnac": "O-GlcNAcylation",
    "o-glycosylation": "O-glycosylation",
    "o-linked-glycosylation": "O-glycosylation",
    "succinylation": "succinylation",
    "acetylation": "acetylation",
    "crotonylation": "crotonylation",
    "malonylation": "malonylation",
    "ubiquitination": "ubiquitination",
    "sumoylation": "sumoylation",
    "phosphorylation": "phosphorylation_ST",  # residue split resolved by step2's chemistry check
    "methylation": "methylation_K",  # K vs R ambiguous at alias stage; step2 corrects R rows
    "s-nitrosylation": "S-nitrosylation",
    "palmitoylation": "palmitoylation",
    "citrullination": "citrullination",
    "nitration": "nitration",
    "sulfation": "sulfation",
    "deamidation": "deamidation",
    "s-palmitoylation": "palmitoylation",
    "glutathionylation": "glutathionylation",
    "sulfhydration": "S-sulfhydration",  # dbPTM's unprefixed name -> cfg's S-prefixed key
    "hydroxylation": "hydroxylation_K",  # CPLM is lysine-only, so its bare name is unambiguous
    "β-hydroxybutyrylation": "beta-hydroxybutyrylation",  # literal Greek beta vs ASCII spelling
    "ubiquitylation": "ubiquitination",  # qPTM's spelling of the same modification
}


# Residue-resolved types: source labels covering MORE THAN ONE chemistry, resolved by looking at
# the actual residue at the site. Checked BEFORE PTM_TYPE_ALIASES and win when the residue is
# known -- see N1_CHANGELOG.md #20 for the measured collapse this prevents.
PTM_TYPE_RESIDUE_RESOLVED: dict[str, dict[str, str]] = {
    "glycosylation": {
        "N": "N-glycosylation",
        "S": "O-glycosylation",
        "T": "O-glycosylation",
    },
    "phosphorylation": {
        "S": "phosphorylation_ST",
        "T": "phosphorylation_ST",
        "Y": "phosphorylation_Y",
    },
    "methylation": {"K": "methylation_K", "R": "methylation_R"},
}


def _normalize_type_key(s: str) -> str:
    return str(s).strip().lower().replace("_", "-").replace("  ", " ").replace(" ", "-")


def normalize_ptm_type(
    raw_type: str, known_types: set | None = None, residue: str | None = None
) -> str:
    """Resolution order: residue-resolved table (only when `residue` is supplied and the label is
    multi-chemistry) -> alias table -> case/formatting variant of an existing canonical key.
    Returns `raw_type` unchanged if nothing matches, so the caller's fail-loud unmapped-type check
    stays exactly as strict as before."""
    key = _normalize_type_key(raw_type)
    if residue and key in PTM_TYPE_RESIDUE_RESOLVED:
        resolved = PTM_TYPE_RESIDUE_RESOLVED[key].get(residue)
        if resolved is not None:
            return resolved
    if key in PTM_TYPE_ALIASES:
        return PTM_TYPE_ALIASES[key]
    if known_types:
        for canonical in known_types:
            if _normalize_type_key(canonical) == key:
                return canonical
    return raw_type


def type_to_stratum(raw_type: str) -> str:
    """Convenience for reporting: canonical stratum for a raw type string, or 'UNMAPPED'."""
    canonical = normalize_ptm_type(raw_type, known_types=set(CFG.ptm_type_to_stratum))
    return CFG.ptm_type_to_stratum.get(canonical, "UNMAPPED")


# ---------------------------------------------------------------------------
# Caching layer, shared by M1/M2/M3. Every cached artefact is paired with a fingerprint sidecar
# (`<file>.fp`), checked before reuse, so a config change can never silently reuse stale output.
#
# NOTE (inherited limitation, not silently fixed): `config_fingerprint` hashes CONFIG, not CODE.
# Editing parsing/filtering logic without changing a Config field will not invalidate the cache --
# bump `force=True` manually after logic edits, exactly as N1's own FORCE_REBUILD flag documents.
# ---------------------------------------------------------------------------
def fingerprint(*parts) -> str:
    """Short stable hash of JSON-serialisable objects; used to invalidate caches."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def find_cached(filename: str, search_dirs: list[Path] | None = None) -> Path | None:
    """Searches every directory in `search_dirs` (recursively) for `filename`, first match wins."""
    for d in search_dirs or []:
        if not d.exists():
            continue
        for match in d.rglob(filename):
            if match.is_file():
                return match
    return None


@dataclass
class CacheResult:
    path: Path
    was_cached: bool
    source: str  # "cache" | "built"


def ensure_artifact(
    filename: str,
    build_fn: Callable[[Path], None],
    *,
    dest_dir: Path,
    search_dirs: list[Path] | None = None,
    expected_fingerprint: str | None = None,
    force: bool = False,
) -> CacheResult:
    """Core cache-or-build primitive."""
    dest = dest_dir / filename

    if not force:
        hit = find_cached(filename, search_dirs=search_dirs or [dest_dir])
        if hit is not None:
            fp_ok = True
            if expected_fingerprint is not None:
                fp_sidecar = hit.with_suffix(hit.suffix + ".fp")
                fp_ok = (
                    fp_sidecar.exists()
                    and fp_sidecar.read_text().strip() == expected_fingerprint
                )
            if fp_ok:
                if hit != dest:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(hit, dest)
                    if expected_fingerprint is not None:
                        dest.with_suffix(dest.suffix + ".fp").write_text(
                            expected_fingerprint
                        )
                return CacheResult(path=dest, was_cached=True, source="cache")
            print(
                f"[cache] {filename}: cached copy found but fingerprint is stale -- rebuilding."
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    build_fn(dest)
    if expected_fingerprint is not None:
        dest.with_suffix(dest.suffix + ".fp").write_text(expected_fingerprint)
    return CacheResult(path=dest, was_cached=False, source="built")


def config_fingerprint(*extra, cfg: Config = None) -> str:
    """Fingerprint of every config field that affects M2/M3's output content. Pass upstream
    artefact hashes (e.g. an M1 manifest digest) as `extra` to also invalidate on raw-input
    changes."""
    cfg = cfg if cfg is not None else CFG
    relevant = {
        "max_protein_length": cfg.max_protein_length,
        "swissprot_reviewed_only": cfg.swissprot_reviewed_only,
        "cdhit_identity": cfg.cdhit_identity,
        "holdout_fraction": cfg.holdout_fraction,
        "holdout_seed": cfg.holdout_seed,
        "holdout_stratify_by": cfg.holdout_stratify_by,
        "min_sites_per_type": cfg.min_sites_per_type,
        "min_sites_headline": cfg.min_sites_headline,
        "ptm_type_to_stratum": cfg.ptm_type_to_stratum,
        "primary_types": cfg.primary_types,
        "stratum_residues": {k: sorted(v) for k, v in cfg.stratum_residues.items()},
        "qptm_reliability_floor": cfg.qptm_reliability_floor,
        "ood_min_sites_per_type": cfg.ood_min_sites_per_type,
    }
    return fingerprint(relevant, *extra)


# ---------------------------------------------------------------------------
# Manifest -- every acquired artefact's provenance (M1's other emission).
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Manifest:
    entries: list[dict] = field(default_factory=list)

    def log(self, source: str, url: str, dest: Path, notes: str = "") -> dict:
        if dest.exists() and dest.is_file():
            sha, size = sha256_file(dest), dest.stat().st_size
        elif dest.exists() and dest.is_dir():
            sha, size = (
                None,
                sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()),
            )
        else:
            sha, size = None, None
        entry = {
            "source": source,
            "url": url,
            "dest": str(dest),
            "sha256": sha,
            "bytes": size,
            "accessed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": notes,
        }
        self.entries.append(entry)
        return entry

    def save(self, path: Path):
        path.write_text(json.dumps(self.entries, indent=2, default=str))
