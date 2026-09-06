"""Adapter for dbPTM benchmark tables."""
from pathlib import Path
from typing import Iterator, Union
from ptm_sae.data.adapters.base import BaseSiteAdapter, canonicalize_ptm_type
from ptm_sae.data.schema import PTMObservation


class DbPTMAdapter(BaseSiteAdapter):
    """
    Parses dbPTM tab-delimited site export tables.
    Standard columns: UniProt_ID, Gene, PTM_Type, Position, Sequence_Window, Source_DB
    """

    def can_handle(self, source: Union[str, Path]) -> bool:
        src_str = str(source).lower()
        return "dbptm" in src_str

    def parse(self, source: Union[str, Path]) -> Iterator[PTMObservation]:
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                return
            headers = [h.strip().lower() for h in header_line.split("\t")]
            
            id_col = next((i for i, h in enumerate(headers) if "id" in h or "acc" in h), 0)
            ptm_col = next((i for i, h in enumerate(headers) if "ptm" in h or "type" in h), 2)
            pos_col = next((i for i, h in enumerate(headers) if "pos" in h or "site" in h), 3)
            flank_col = next((i for i, h in enumerate(headers) if "flank" in h or "seq" in h or "window" in h), None)

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) <= max(id_col, ptm_col, pos_col):
                    continue

                u_id = parts[id_col].strip()
                ptm_raw = parts[ptm_col].strip()
                pos_str = parts[pos_col].strip()

                try:
                    pos = int(pos_str)
                except ValueError:
                    continue

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
                    flanking_15mer=flank,
                )
