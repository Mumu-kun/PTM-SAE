"""PTM Modular SAE Engine.

The heavy entry points load on first use, not at `import ptm_sae`: `import ptm_sae.runtime` must stay
standard-library only, because the notebooks import it *before* installing (and possibly upgrading)
numpy/pandas/torch. A package imported before pip upgrades it stays half-old inside the running
kernel and later imports fail (observed on Kaggle: numpy could not import its own submodules).
"""

import importlib

__version__ = "0.1.0"

_LAZY = {
    "run_full_lifecycle": "ptm_sae.pipeline",
    "run_sae_training": "ptm_sae.training.train",
}
__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'ptm_sae' has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY[name]), name)
