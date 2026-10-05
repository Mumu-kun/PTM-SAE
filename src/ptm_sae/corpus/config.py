"""Paths and the Config the corpus maintenance code (verify / resplit / audit) reads.

Ported from the partner's `docs/reference/N1.ipynb` (cell 3, "M0 -- Configuration & Design
Invariants"), trimmed to what is still used after the corpus build was archived (branch
`archive/corpus-build`: acquisition, label cascade, type canonicalisation, cache/fingerprint scaffold).
The Kaggle-path branching (`/kaggle/working`, `/kaggle/input`, `/kaggle/temp`) is removed entirely --
`CorpusPaths` is env-var driven instead, with no Kaggle/local flag.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from ptm_sae.data.splitting import SplitSettings
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
# Config -- the fields verify / resplit / the audit read.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Config:
    # Headline bar: sites a primary PTM type needs in discovery (verify_outputs)
    min_sites_headline: int = 1000

    # Redundancy -- 40% identity, per the user's decision (see the implementation plan).
    cdhit_identity: float = 0.40

    # Final 3-way partition (data/splitting.py): fractions, per-type site floors, homology audit.
    split: SplitSettings = dataclasses.field(default_factory=SplitSettings)

    # Residue strata -> chemistry-valid target residues. Codes match N1 exactly: K, ST,
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
    # The three primary/headline types this thesis leads with.
    primary_types: tuple = ("N-glycosylation", "O-GlcNAcylation", "succinylation")


CFG = Config()
