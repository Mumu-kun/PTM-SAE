"""Adapter for PhosphoSitePlus export datasets."""
from pathlib import Path
import re
from typing import Iterator, Union
from ptm_sae.data.adapters.base import BaseSiteAdapter, canonicalize_ptm_type
from ptm_sae.data.schema import PTMObservation


class PhosphoSitePlusAdapter(BaseSiteAdapter):
    """
    Parses PhosphoSitePlus dataset files (Phosphorylation, Ubiquitination, Acetylation, etc.).
    Skips comment lines (#), filters for human species, parses MOD_RSD (e.g. S392-p -> S, 392).
    """

    def can_handle(self, source: Union[str, Path]) -> bool:
        src_str = str(source).lower()
        return "phosphosite" in src_str or "site_dataset" in src_str

    def parse(self, source: Union[str, Path]) -> Iterator[PTMObservation]:
        path = Path(source)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            headers = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if headers is None:
                    headers = [h.strip().upper() for h in line.split("\t")]
                    acc_idx = headers.index("ACC_ID") if "ACC_ID" in headers else 1
                    mod_idx = headers.index("MOD_RSD") if "MOD_RSD" in headers else 4
                    org_idx = headers.index("ORGANISM") if "ORGANISM" in headers else 6
                    continue

                parts = line.split("\t")
                if len(parts) <= max(acc_idx, mod_idx, org_idx):
                    continue

                organism = parts[org_idx].strip().lower()
                if organism != "human":
                    continue

                u_id = parts[acc_idx].strip()
                mod_rsd = parts[mod_idx].strip()  # e.g. 'S15-p' or 'K382-ub'

                match = re.search(r"^([A-Z])(\d+)", mod_rsd)
                if not match:
                    continue

                residue = match.group(1)
                pos = int(match.group(2))

                # Extract modification type
                ptm_type = "Phosphorylation"
                if "-ub" in mod_rsd.lower():
                    ptm_type = "Ubiquitination"
                elif "-ac" in mod_rsd.lower():
                    ptm_type = "Acetylation"
                elif "-m" in mod_rsd.lower():
                    ptm_type = "Methylation"

                yield PTMObservation(
                    source_db="PhosphoSitePlus",
                    uniprot_id=u_id,
                    position=pos,
                    residue=residue,
                    canonical_ptm_type=ptm_type,
                    raw_ptm_name=mod_rsd,
                    evidence_tier="experimental",
                )
