"""Lightweight training-time canary for the Residue-Dominance Gate (an explicit Phase 2
acceptance metric in `proposal/Unified_Thesis_Plan_Modular_SAE_PTM.md`, mirroring Member 2's
M9/M16): a latent is "residue-dominant" if a large share of its top-activating positions on
`discovery_val` share one amino acid identity, i.e. it behaves like a generic residue-identity
detector (Residue Collapse) rather than tracking anything more specific. Unlike
`collapse_check.py`, this needs no PTM labels at all — only residue identity, already available
from `proteins.jsonl`'s sequences — so it's cheaper and carries no held_out risk by construction.
"""

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from ptm_sae.extraction.hub import resolve_hf_token, retry_with_backoff
from ptm_sae.extraction.reader import SafeTensorsReader

# Standard 20 amino acids; anything else (rare/ambiguous codes) falls into one shared "unknown"
# bucket rather than growing the vocabulary per oddity in the sequence data.
_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INDEX = {aa: i for i, aa in enumerate(_AA_VOCAB)}
_UNKNOWN_INDEX = len(_AA_VOCAB)
NUM_RESIDUE_CLASSES = len(_AA_VOCAB) + 1


def load_discovery_val_sequences(
    corpus_dir: str | Path,
    remote_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> dict[str, str]:
    """Loads `proteins.jsonl`, filtered to `discovery_val`, keyed by `uniprot_id -> sequence`.
    Mirrors `training.dataset.load_partition_ids`'s local-cache-then-remote-hydrate pattern."""
    local_path = Path(corpus_dir) / "proteins.jsonl"

    if not local_path.exists() and remote_repo_id:
        resolved_token = resolve_hf_token(token)
        remote_path = f"{remote_corpus_subpath.rstrip('/')}/proteins.jsonl"

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
            f"proteins.jsonl not found locally at {local_path} or on remote repository "
            f"{remote_repo_id}."
        )

    sequences: dict[str, str] = {}
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("partition") == "discovery_val":
                sequences[record["uniprot_id"]] = record["sequence"]
    return sequences


class ResidueDominanceAccumulator:
    """Streaming per-latent top-k activation tracker. For each latent, keeps the k highest
    activation values seen so far and the amino-acid identity of the residue each came from."""

    def __init__(self, d_hidden: int, top_k: int, device: torch.device):
        self.top_k = top_k
        self.values = torch.full((d_hidden, top_k), float("-inf"), device=device)
        self.residue_codes = torch.full((d_hidden, top_k), -1, dtype=torch.long, device=device)

    def update(self, latents: torch.Tensor, residue_codes: torch.Tensor) -> None:
        """`latents`: (L, d_hidden) raw activation values. `residue_codes`: (L,) long, amino-acid
        vocabulary index (0..NUM_RESIDUE_CLASSES-1)."""
        combined_values = torch.cat([self.values, latents.t()], dim=1)  # (d_hidden, top_k + L)
        combined_codes = torch.cat(
            [self.residue_codes, residue_codes.unsqueeze(0).expand(self.values.shape[0], -1)], dim=1
        )
        new_values, idx = torch.topk(combined_values, self.top_k, dim=1)
        self.values = new_values
        self.residue_codes = torch.gather(combined_codes, 1, idx)

    def finalize(self, threshold: float) -> dict[str, float]:
        valid = self.residue_codes >= 0
        one_hot = torch.nn.functional.one_hot(self.residue_codes.clamp_min(0), NUM_RESIDUE_CLASSES)
        one_hot = one_hot * valid.unsqueeze(-1)
        counts = one_hot.sum(dim=1).float()  # (d_hidden, NUM_RESIDUE_CLASSES)
        valid_counts = valid.sum(dim=1).clamp_min(1).float()  # (d_hidden,)

        dominance_frac = counts.max(dim=1).values / valid_counts
        is_dominant = dominance_frac >= threshold
        return {
            "collapse_rate": is_dominant.float().mean().item(),
            "n_latents": float(self.values.shape[0]),
        }


@torch.no_grad()
def run_residue_dominance_check(
    model,
    sequences: dict[str, str],
    config,
    device: torch.device,
    top_k: int,
    threshold: float,
) -> dict[str, float]:
    """One pass over discovery_val proteins, read directly via SafeTensorsReader (bypassing the
    shuffled training DataLoader — order doesn't matter here)."""
    model.eval()
    reader = SafeTensorsReader(
        cache_dir=config.cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=config.remote_subpath,
        max_cached_shards=config.max_cached_shards,
    )
    accumulator = ResidueDominanceAccumulator(config.d_hidden, top_k, device)
    try:
        for uniprot_id, sequence in sequences.items():
            if uniprot_id not in reader.entries:
                continue
            activations = reader.get_protein_activations(uniprot_id).to(device)
            # Guards against any off-by-one between the cached sequence and the activation
            # shard rather than assuming they always agree exactly.
            length = min(activations.shape[0], len(sequence))
            if length == 0:
                continue
            activations = activations[:length]
            residue_codes = torch.tensor(
                [_AA_TO_INDEX.get(aa, _UNKNOWN_INDEX) for aa in sequence[:length]],
                dtype=torch.long,
                device=device,
            )

            out = model(activations)
            accumulator.update(out.latents, residue_codes)
    finally:
        reader.close()
    model.train()
    return accumulator.finalize(threshold)
