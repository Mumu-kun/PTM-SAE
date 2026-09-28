"""N1-derived PTM corpus construction: acquisition (M1) -> clustering (M2) -> label cascade (M3).

Ported from the partner's `src/ptm_eval/N1.ipynb`. See `corpus/README.md` for the `cd-hit`
system-binary dependency and `corpus/pipeline.py` for the runnable end-to-end orchestrator.
"""

from ptm_sae.corpus.config import CFG, Config, CorpusPaths

__all__ = ["CFG", "Config", "CorpusPaths"]
