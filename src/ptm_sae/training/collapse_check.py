"""Lightweight training-time canary for Residue Collapse (see `CONTEXT.md`).

This is deliberately not Member 2's M1-M25 statistical evaluation pipeline — no contingency
tables, no Fisher's exact test, no permutation nulls. It's a coarse, cheap signal computed
directly on `discovery_val` (never `held_out`) to decide whether a checkpoint is even worth
handing off to that full evaluation, along three axes:

  - by_stratum: does any latent fire on modified residues at a rate elevated over that
    stratum's own PTM base rate, or does every latent track the base rate everywhere (i.e.,
    latents behave like generic residue-identity detectors — Residue Collapse)?
  - by_ptm_type: within a stratum, is a latent's elevated firing specific to ONE PTM type, or
    just "any modification" (still a form of collapse, one level more specific)? Also reports
    a selectivity spot-check — one clear winning latent per PTM type, or several mediocre,
    undisentangled ones.
  - by_negative_tier: does a latent's apparent PTM-association hold up against curated `hard`
    negatives (residues chosen to closely resemble modified ones), or does it fire on those
    almost as often as on real modifications — a shortcut-learning flag, not real chemistry?
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from ptm_sae.extraction.hub import resolve_hf_token, retry_with_backoff
from ptm_sae.extraction.reader import SafeTensorsReader

# A latent firing on modified residues at 2x a stratum's own base rate is treated as a
# candidate non-generic (PTM-associated) feature worth flagging — a heuristic canary
# threshold, not a statistical significance test (that's Member 2's job).
BASELINE_RATIO_THRESHOLD = 2.0
# A latent within 80% of a PTM type's best concentration ratio is treated as "tied" with it
# for the selectivity spot-check.
NEAR_BEST_RATIO_THRESHOLD = 0.8
# A latent firing on `hard` negatives at >80% of its modified-residue firing rate can't
# actually tell the two apart — a shortcut-learning flag.
SHORTCUT_ACTIVITY_RATIO_THRESHOLD = 0.8


@dataclass(frozen=True)
class ResidueLabel:
    stratum: str
    ptm_types: frozenset[str]  # empty == unmodified
    negative_tier: str | None  # "verified" / "hard" / "background" for negatives, else None


StratumLabels = dict[str, dict[int, ResidueLabel]]


def load_discovery_val_labels(
    corpus_dir: str | Path,
    remote_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> StratumLabels:
    """Loads `ptm_sites.jsonl`, filtered to `discovery_val`, keyed by
    `uniprot_id -> {position: ResidueLabel}`. Mirrors `training.dataset.load_partition_ids`'s
    local-cache-then-remote-hydrate pattern.
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
            labels.setdefault(record["uniprot_id"], {})[record["position"]] = ResidueLabel(
                stratum=record["stratum"],
                ptm_types=frozenset(record.get("ptm_types") or []),
                negative_tier=record.get("negative_tier"),
            )
    return labels


class PTMConcentrationAccumulator:
    """Streaming per-latent activation/PTM-label concentration accumulator, tracked along
    three parallel breakdowns (stratum, (stratum, ptm_type) pair, (stratum, negative_tier)
    pair) — all sharing the per-stratum totals as their common denominator.
    """

    def __init__(
        self,
        strata: list[str],
        ptm_pairs: list[tuple[str, str]],
        negtier_pairs: list[tuple[str, str]],
        d_hidden: int,
        device: torch.device,
    ):
        self.strata = strata
        self.stratum_to_id = {s: i for i, s in enumerate(strata)}
        n_strata = len(strata)
        self.total_count = torch.zeros(n_strata, device=device)
        self.total_modified_count = torch.zeros(n_strata, device=device)
        self.active_count = torch.zeros(n_strata, d_hidden, device=device)
        self.active_modified_count = torch.zeros(n_strata, d_hidden, device=device)

        self.ptm_pairs = ptm_pairs
        self.ptm_pair_to_id = {p: i for i, p in enumerate(ptm_pairs)}
        n_pairs = len(ptm_pairs)
        self.total_positive_count = torch.zeros(n_pairs, device=device)
        self.active_positive_count = torch.zeros(n_pairs, d_hidden, device=device)

        self.negtier_pairs = negtier_pairs
        self.negtier_pair_to_id = {p: i for i, p in enumerate(negtier_pairs)}
        n_negtier = len(negtier_pairs)
        self.total_negtier_count = torch.zeros(n_negtier, device=device)
        self.active_negtier_count = torch.zeros(n_negtier, d_hidden, device=device)

    def update_stratum(
        self, active: torch.Tensor, stratum_id: torch.Tensor, is_modified: torch.Tensor
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

    def update_ptm_pairs(
        self, active: torch.Tensor, pair_ids: torch.Tensor, residue_idx: torch.Tensor
    ) -> None:
        """A residue can belong to several (stratum, ptm_type) pairs at once (multi-label PTM
        crosstalk), so membership is given as flat (pair_id, residue_idx) correspondences
        rather than a single per-residue id."""
        if pair_ids.numel() == 0:
            return
        sel_active = active[residue_idx].float()
        ones = torch.ones(pair_ids.shape[0], device=pair_ids.device)
        self.total_positive_count.index_add_(0, pair_ids, ones)
        self.active_positive_count.index_add_(0, pair_ids, sel_active)

    def update_negative_tier(self, active: torch.Tensor, negtier_id: torch.Tensor) -> None:
        """`negtier_id`: (L,) long, -1 for residues with no negative_tier (positives)."""
        mask = negtier_id >= 0
        if not mask.any():
            return
        sel_id = negtier_id[mask]
        sel_active = active[mask].float()
        ones = torch.ones_like(sel_id, dtype=torch.float)
        self.total_negtier_count.index_add_(0, sel_id, ones)
        self.active_negtier_count.index_add_(0, sel_id, sel_active)

    def finalize_stratum(self) -> dict[str, dict[str, float]]:
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

    def finalize_ptm_type(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for p, (stratum, ptm_type) in enumerate(self.ptm_pairs):
            stratum_idx = self.stratum_to_id[stratum]
            total = max(1.0, self.total_count[stratum_idx].item())
            base_rate = self.total_positive_count[p].item() / total
            has_activity = self.active_count[stratum_idx] > 0
            key = f"{stratum}__{ptm_type}"

            if base_rate <= 0 or not bool(has_activity.any()):
                results[key] = {
                    "mean_concentration_ratio": 0.0,
                    "n_latents_above_2x_baseline": 0.0,
                    "best_latent_ratio": 0.0,
                    "n_latents_near_best": 0.0,
                }
                continue

            active_positive_rate = (
                self.active_positive_count[p, has_activity] / self.active_count[stratum_idx, has_activity]
            )
            ratio = active_positive_rate / base_rate
            best_ratio = ratio.max().item()
            n_near_best = (
                float((ratio > NEAR_BEST_RATIO_THRESHOLD * best_ratio).sum().item()) if best_ratio > 0 else 0.0
            )
            results[key] = {
                "mean_concentration_ratio": ratio.mean().item(),
                "n_latents_above_2x_baseline": float((ratio > BASELINE_RATIO_THRESHOLD).sum().item()),
                "best_latent_ratio": best_ratio,
                "n_latents_near_best": n_near_best,
            }
        return results

    def finalize_negative_tier(self) -> dict[str, dict[str, float]]:
        """Only `hard` negatives produce a shortcut-suspect flag — `verified`/`background`
        negatives aren't curated to resemble modified residues, so a latent distinguishing
        those from real modifications proves nothing about shortcut learning."""
        results: dict[str, dict[str, float]] = {}
        for t, (stratum, tier) in enumerate(self.negtier_pairs):
            if tier != "hard":
                continue
            stratum_idx = self.stratum_to_id[stratum]
            total = max(1.0, self.total_count[stratum_idx].item())
            base_rate = self.total_modified_count[stratum_idx].item() / total
            has_activity = self.active_count[stratum_idx] > 0

            if base_rate <= 0 or not bool(has_activity.any()):
                results[stratum] = {"n_latents_shortcut_suspect": 0.0}
                continue

            active_modified_rate = (
                self.active_modified_count[stratum_idx, has_activity] / self.active_count[stratum_idx, has_activity]
            )
            concentration_ratio = active_modified_rate / base_rate
            high_concentration = concentration_ratio > BASELINE_RATIO_THRESHOLD

            total_hard = max(1.0, self.total_negtier_count[t].item())
            total_modified = max(1.0, self.total_modified_count[stratum_idx].item())
            rate_hard = self.active_negtier_count[t, has_activity] / total_hard
            rate_modified = self.active_modified_count[stratum_idx, has_activity] / total_modified
            cant_tell_apart = (rate_modified > 0) & (
                (rate_hard / rate_modified.clamp_min(1e-8)) > SHORTCUT_ACTIVITY_RATIO_THRESHOLD
            )

            results[stratum] = {
                "n_latents_shortcut_suspect": float((high_concentration & cant_tell_apart).sum().item())
            }
        return results


@torch.no_grad()
def run_ptm_concentration_check(
    model, labels: StratumLabels, config, device: torch.device
) -> dict[str, dict[str, dict[str, float]]]:
    """One pass over labeled discovery_val proteins, read directly via SafeTensorsReader
    (bypassing the shuffled training DataLoader — order doesn't matter here). Returns
    `{"by_stratum": ..., "by_ptm_type": ..., "by_negative_tier": ...}`."""
    all_labels = [label for positions in labels.values() for label in positions.values()]
    strata = sorted({label.stratum for label in all_labels})
    if not strata:
        return {"by_stratum": {}, "by_ptm_type": {}, "by_negative_tier": {}}

    ptm_pairs = sorted({(label.stratum, ptm_type) for label in all_labels for ptm_type in label.ptm_types})
    negtier_pairs = sorted(
        {(label.stratum, label.negative_tier) for label in all_labels if label.negative_tier is not None}
    )

    model.eval()
    reader = SafeTensorsReader(
        cache_dir=config.cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=config.remote_subpath,
        max_cached_shards=config.max_cached_shards,
    )
    accumulator = PTMConcentrationAccumulator(strata, ptm_pairs, negtier_pairs, config.d_hidden, device)
    try:
        for uniprot_id, position_labels in labels.items():
            if uniprot_id not in reader.entries:
                continue
            activations = reader.get_protein_activations(uniprot_id).to(device)
            length = activations.shape[0]

            stratum_id = torch.full((length,), -1, dtype=torch.long, device=device)
            is_modified = torch.zeros(length, dtype=torch.bool, device=device)
            negtier_id = torch.full((length,), -1, dtype=torch.long, device=device)
            pair_ids: list[int] = []
            pair_residue_idx: list[int] = []

            for position, label in position_labels.items():
                if not (1 <= position <= length):
                    continue
                idx = position - 1
                stratum_id[idx] = accumulator.stratum_to_id[label.stratum]
                is_modified[idx] = bool(label.ptm_types)
                if label.negative_tier is not None:
                    negtier_id[idx] = accumulator.negtier_pair_to_id[(label.stratum, label.negative_tier)]
                for ptm_type in label.ptm_types:
                    pair_ids.append(accumulator.ptm_pair_to_id[(label.stratum, ptm_type)])
                    pair_residue_idx.append(idx)

            out = model(activations)
            active = out.latents.gt(0)
            accumulator.update_stratum(active, stratum_id, is_modified)
            accumulator.update_negative_tier(active, negtier_id)
            accumulator.update_ptm_pairs(
                active,
                torch.tensor(pair_ids, dtype=torch.long, device=device),
                torch.tensor(pair_residue_idx, dtype=torch.long, device=device),
            )
    finally:
        reader.close()
    model.train()
    return {
        "by_stratum": accumulator.finalize_stratum(),
        "by_ptm_type": accumulator.finalize_ptm_type(),
        "by_negative_tier": accumulator.finalize_negative_tier(),
    }
