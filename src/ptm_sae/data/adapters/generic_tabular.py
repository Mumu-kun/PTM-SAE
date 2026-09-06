"""Generic configurable tabular adapter for CSV/TSV data."""
from pathlib import Path
from typing import Dict, Iterator, Optional, Union
from ptm_sae.data.adapters.base import BaseSiteAdapter, canonicalize_ptm_type
from ptm_sae.data.schema import PTMObservation


class GenericTabularAdapter(BaseSiteAdapter):
    """
    Configurable tabular adapter with custom column name mappings.
    Useful for CPLM, qPTM, O-GlcNAcAtlas, or arbitrary in-house TSV files.
    """

    def __init__(
        self,
        source_name: str = "Generic",
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = "\t",
    ):
        self.source_name = source_name
        self.col_map = column_mapping or {
            "uniprot_id": "uniprot_id",
            "position": "position",
            "residue": "residue",
            "ptm_type": "ptm_type",
        }
        self.delimiter = delimiter

    def can_handle(self, source: Union[str, Path]) -> bool:
        src_str = str(source).lower()
        return src_str.endswith(".tsv") or src_str.endswith(".csv") or src_str.endswith(".tab")

    def parse(self, source: Union[str, Path]) -> Iterator[PTMObservation]:
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                return
            headers = [h.strip().lower() for h in header_line.split(self.delimiter)]

            id_col = headers.index(self.col_map["uniprot_id"].lower())
            pos_col = headers.index(self.col_map["position"].lower())
            res_col = headers.index(self.col_map["residue"].lower()) if "residue" in self.col_map and self.col_map["residue"].lower() in headers else None
            type_col = headers.index(self.col_map["ptm_type"].lower()) if "ptm_type" in self.col_map and self.col_map["ptm_type"].lower() in headers else None

            for line in f:
                parts = line.strip().split(self.delimiter)
                if len(parts) <= max(id_col, pos_col):
                    continue

                u_id = parts[id_col].strip()
                try:
                    pos = int(parts[pos_col].strip())
                except ValueError:
                    continue

                res = parts[res_col].strip().upper() if res_col is not None and len(parts) > res_col else "S"
                ptm_raw = parts[type_col].strip() if type_col is not None and len(parts) > type_col else "Phosphorylation"

                yield PTMObservation(
                    source_db=self.source_name,
                    uniprot_id=u_id,
                    position=pos,
                    residue=res,
                    canonical_ptm_type=canonicalize_ptm_type(ptm_raw),
                    raw_ptm_name=ptm_raw,
                    evidence_tier="curated",
                )
