"""Lightweight training-time canary for Residue Collapse (see `CONTEXT.md`).

This is deliberately not Member 2's M1-M25 statistical evaluation pipeline — no contingency
tables, no Fisher's exact test, no permutation nulls. It's a coarse, cheap signal computed
directly on `discovery_val` (never `held_out`) to decide whether a checkpoint is even worth
handing off to that full evaluation: for each chemical stratum, does any latent fire on modified
residues at a rate elevated over that stratum's own PTM base rate, or does every latent's firing
rate track the base rate everywhere (i.e., latents are behaving like generic residue-identity
detectors — Residue Collapse)?
"""

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from ptm_sae.extraction.hub import resolve_hf_token, retry_with_backoff
from ptm_sae.extraction.reader import SafeTensorsReader

# A latent firing on modified residues at 2x a stratum's own base rate is treated as a
# candidate non-generic (PTM-associated) feature worth flagging — a heuristic canary
# threshold, not a statistical significance test (that's Member 2's job).
BASELINE_RATIO_THRESHOLD = 2.0

StratumLabels = dict[str, dict[int, tuple[str, bool]]]


def load_discovery_val_labels(
    corpus_dir: str | Path,
    remote_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> StratumLabels:
    """Loads `ptm_sites.jsonl`, filtered to `discovery_val`, keyed by
    `uniprot_id -> {position: (stratum, is_modified)}`. Mirrors
    `training.dataset.load_partition_ids`'s local-cache-then-remote-hydrate pattern.
    """
    local_path = Path(corpus_dir) / "ptm_sites.jsonl"

    if not local_path.exists() and remote_repo_id:
        resolved_token = resolve_hf_token(token)
        remote_path = f"{remote_corpus_subpath.rstrip('/')}/ptm_sites.jsonl"

        def _download() -> str:
            return hf_hub_download(
                repo_id=remote_repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=resolved_token,
            )

        cached_file = retry_with_backoff(_download, max_retries=3, base_delay=2.0)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"ptm_sites.jsonl not found locally at {local_path} or on remote repository "
            f"{remote_repo_id}."
        )

    labels: StratumLabels = {}
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("partition") != "discovery_val":
                continue
            is_modified = bool(record.get("ptm_types"))
            labels.setdefault(record["uniprot_id"], {})[record["position"]] = (
                record["stratum"],
                is_modified,
            )
    return labels


class CollapseCheckAccumulator:
    """Streaming per-stratum, per-latent activation/PTM-label concentration accumulator.

    For each stratum, tracks (a) how often each latent fires on any residue in that stratum,
    and (b) how often it fires specifically on *modified* residues in that stratum — the ratio
    of (b)/(a) against the stratum's overall modified fraction is the concentration signal.
    """

    def __init__(self, strata: list[str], d_hidden: int, device: torch.device):
        self.strata = strata
        self.stratum_to_id = {s: i for i, s in enumerate(strata)}
        num_strata = len(strata)
        self.total_count = torch.zeros(num_strata, device=device)
        self.total_modified_count = torch.zeros(num_strata, device=device)
        self.active_count = torch.zeros(num_strata, d_hidden, device=device)
        self.active_modified_count = torch.zeros(num_strata, d_hidden, device=device)

    def update(
        self,
        active: torch.Tensor,
        stratum_id: torch.Tensor,
        is_modified: torch.Tensor,
    ) -> None:
        """`active`: (L, d_hidden) bool. `stratum_id`: (L,) long, -1 for unlabeled residues.
        `is_modified`: (L,) bool."""
        mask = stratum_id >= 0
        if not mask.any():
            return
        sel_stratum = stratum_id[mask]
        sel_active = active[mask].float()
        sel_modified = is_modified[mask]
        ones = torch.ones_like(sel_stratum, dtype=torch.float)

        self.total_count.index_add_(0, sel_stratum, ones)
        self.active_count.index_add_(0, sel_stratum, sel_active)
        if sel_modified.any():
            self.total_modified_count.index_add_(0, sel_stratum[sel_modified], ones[sel_modified])
            self.active_modified_count.index_add_(0, sel_stratum[sel_modified], sel_active[sel_modified])

    def finalize(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for i, stratum in enumerate(self.strata):
            total = max(1.0, self.total_count[i].item())
            base_rate = self.total_modified_count[i].item() / total
            has_activity = self.active_count[i] > 0

            if base_rate <= 0 or not bool(has_activity.any()):
                results[stratum] = {
                    "mean_concentration_ratio": 0.0,
                    "n_latents_above_2x_baseline": 0.0,
                    "n_active_latents": float(has_activity.sum().item()),
                }
                continue

            active_modified_rate = (
                self.active_modified_count[i, has_activity] / self.active_count[i, has_activity]
            )
            ratio = active_modified_rate / base_rate
            results[stratum] = {
                "mean_concentration_ratio": ratio.mean().item(),
                "n_latents_above_2x_baseline": float((ratio > BASELINE_RATIO_THRESHOLD).sum().item()),
                "n_active_latents": float(has_activity.sum().item()),
            }
        return results


@torch.no_grad()
def run_collapse_check(model, labels: StratumLabels, config, device: torch.device) -> dict[str, dict[str, float]]:
    """One pass over labeled discovery_val proteins, read directly via SafeTensorsReader
    (bypassing the shuffled training DataLoader — order doesn't matter here)."""
    strata = sorted({stratum for positions in labels.values() for stratum, _ in positions.values()})
    if not strata:
        return {}

    model.eval()
    reader = SafeTensorsReader(
        cache_dir=config.cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=config.remote_subpath,
        max_cached_shards=config.max_cached_shards,
    )
    accumulator = CollapseCheckAccumulator(strata, config.d_hidden, device)
    try:
        for uniprot_id, position_labels in labels.items():
            if uniprot_id not in reader.entries:
                continue
            activations = reader.get_protein_activations(uniprot_id).to(device)
            length = activations.shape[0]

            stratum_id = torch.full((length,), -1, dtype=torch.long, device=device)
            is_modified = torch.zeros(length, dtype=torch.bool, device=device)
            for position, (stratum, modified) in position_labels.items():
                if 1 <= position <= length:
                    stratum_id[position - 1] = accumulator.stratum_to_id[stratum]
                    is_modified[position - 1] = modified

            out = model(activations)
            accumulator.update(out.latents.gt(0), stratum_id, is_modified)
    finally:
        reader.close()
    model.train()
    return accumulator.finalize()
