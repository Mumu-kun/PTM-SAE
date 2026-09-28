"""Retrieval helpers for pulling a training run's logged data back out of W&B.

Three distinct payload shapes get logged by `ptm_sae.training.train`, each with a retrieval
gotcha worth solving once: plain scalars (`train/*`, `val/*`), a `wandb.Histogram` (the final
feature-density histogram), and a `wandb.Table` (the collapse-check summary). Pure retrieval —
no plotting, no pass/fail judgment; those live in `ptm_sae.analysis.diagnostics`.
"""

import numpy as np
import pandas as pd
import wandb


def fetch_run(project: str, run_id: str) -> wandb.apis.public.Run:
    return wandb.Api().run(f"{project}/{run_id}")


def pull_scalars(run: wandb.apis.public.Run, keys: list[str]) -> pd.DataFrame:
    """Pulls scalar history for `keys` into a DataFrame indexed by step.

    Uses `scan_history` (not `history`), which returns every logged row rather than a
    subsampled slice — `history()`'s default subsampling would smear out exactly the kind of
    early-step transitions (e.g. a threshold starting to diverge) this analysis needs to see
    clearly. Callers should pull metrics logged at different cadences (e.g. `train/*` every 50
    steps vs. `val/*` every 500) in separate calls, since combining mismatched-cadence keys in
    one `scan_history` call forces a misleading forward-fill/NaN alignment across rows.
    """
    rows = list(run.scan_history(keys=["_step", *keys]))
    return pd.DataFrame(rows).set_index("_step").sort_index()


def pull_histogram(
    run: wandb.apis.public.Run, key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (counts, bin_edges) for a `wandb.Histogram`-typed value, from its final logged
    row. The API hands this back as a dict with "values"/"bins" (not a plain scalar column);
    for uniform-width bins (the common case here) "bins" is None and the edges are instead
    packed as {"min", "size", "count"} under "packedBins", which this reconstructs."""
    rows = list(run.scan_history(keys=["_step", key]))
    payload = rows[-1][key]
    counts = np.asarray(payload["values"])
    if payload["bins"] is not None:
        return counts, np.asarray(payload["bins"])
    packed = payload["packedBins"]
    edges = packed["min"] + packed["size"] * np.arange(packed["count"] + 1)
    return counts, edges


def pull_table(
    run: wandb.apis.public.Run, artifact_name: str, table_key: str
) -> pd.DataFrame:
    """Downloads a `wandb.Table`-typed value logged under `table_key`, via the run artifact
    it was saved as (tables aren't retrievable through `history()`/`scan_history()`)."""
    for artifact in run.logged_artifacts():
        if artifact.name.startswith(artifact_name):
            table = artifact.get(table_key)
            return table.get_dataframe()
    raise ValueError(f"No artifact named '{artifact_name}' found on run {run.id}")
