"""Maintenance of the published PTM corpus: paths/config, the homology-aware three-way partition and its
cd-hit-2d audit, `verify_outputs` / `resplit` / `upload_to_hf`. The corpus build itself is archived on
the git branch `archive/corpus-build`; see `corpus/README.md`.
"""

from ptm_sae.corpus.config import CFG, Config, CorpusPaths

__all__ = ["CFG", "Config", "CorpusPaths"]
