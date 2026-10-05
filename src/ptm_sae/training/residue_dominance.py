"""Lightweight training-time canary for the Residue-Dominance Gate (an explicit Phase 2
acceptance metric in `proposal/Unified_Thesis_Plan_Modular_SAE_PTM.md`, mirroring Member 2's
M9/M16): a latent is "residue-dominant" if a large share of its top-activating positions on
`discovery_val` share one amino acid identity, i.e. it behaves like a generic residue-identity
detector (Residue Collapse) rather than tracking anything more specific. Unlike
`collapse_check.py`, this needs no PTM labels at all — only residue identity, already available
from `corpus.parquet`'s sequences — so it's cheaper and carries no held_out risk by construction.
"""

from pathlib import Path

import pyarrow.parquet as pq
import torch

from ptm_sae.extraction.reader import SafeTensorsReader
from ptm_sae.training.dataset import hydrate_corpus_file, partition_root

# Standard 20 amino acids; anything else (rare/ambiguous codes) falls into one shared "unknown"
# bucket rather than growing the vocabulary per oddity in the sequence data.
_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INDEX = {aa: i for i, aa in enumerate(_AA_VOCAB)}
_UNKNOWN_INDEX = len(_AA_VOCAB)
NUM_RESIDUE_CLASSES = len(_AA_VOCAB) + 1


def load_discovery_val_sequences(
    corpus_dir: str | Path,
    remote_corpus_repo_id: str | None = None,
    remote_corpus_subpath: str = "corpus",
    token: str | None = None,
) -> dict[str, str]:
    """Loads `corpus.parquet`, filtered to `discovery_val`, keyed by `uniprot_id -> sequence`."""
    corpus_path = hydrate_corpus_file(
        "corpus.parquet", corpus_dir, remote_corpus_repo_id, remote_corpus_subpath, token
    )
    table = pq.read_table(
        corpus_path, columns=["uniprot_id", "sequence"], filters=[("partition", "==", "discovery_val")]
    )
    return {record["uniprot_id"]: record["sequence"] for record in table.to_pylist()}


class ResidueDominanceAccumulator:
    """Streaming per-latent top-k activation tracker. For each latent, keeps the k highest
    activation values seen so far and the amino-acid identity of the residue each came from."""

    def __init__(self, d_hidden: int, top_k: int, device: torch.device):
        self.top_k = top_k
        self.values = torch.full((d_hidden, top_k), float("-inf"), device=device)
        self.residue_codes = torch.full(
            (d_hidden, top_k), -1, dtype=torch.long, device=device
        )
        self.residue_counts = torch.zeros(NUM_RESIDUE_CLASSES, dtype=torch.long, device=device)

    def update(self, latents: torch.Tensor, residue_codes: torch.Tensor) -> None:
        """`latents`: (L, d_hidden) raw activation values. `residue_codes`: (L,) long, amino-acid
        vocabulary index (0..NUM_RESIDUE_CLASSES-1)."""
        self.residue_counts += torch.bincount(residue_codes, minlength=NUM_RESIDUE_CLASSES)
        combined_values = torch.cat(
            [self.values, latents.t()], dim=1
        )  # (d_hidden, top_k + L)
        combined_codes = torch.cat(
            [
                self.residue_codes,
                residue_codes.unsqueeze(0).expand(self.values.shape[0], -1),
            ],
            dim=1,
        )
        new_values, idx = torch.topk(combined_values, self.top_k, dim=1)
        self.values = new_values
        self.residue_codes = torch.gather(combined_codes, 1, idx)

    def finalize(self, threshold: float) -> dict[str, float]:
        """Dominance statistics over the latents that can be scored at all.

        Only POSITIVE activations count: a TopK latent is exactly zero when off, and such slots carry arbitrary
        residues (ties are broken arbitrarily). A latent is *alive* (scored) once it has a full top-k of positive
        activations; the others are not counted as "not collapsed", they are left out, so dead latents cannot
        make a run look better. `collapse_rate` keeps the all-latents denominator for continuity; the primary
        number is `collapse_rate_alive`, with the rate at the neighbouring thresholds (the 0.7 / top-10 setting is
        a heuristic) and `chance_rate`, what independent draws from the observed amino-acid frequencies would give."""
        valid = (self.residue_codes >= 0) & (self.values > 0)
        one_hot = torch.nn.functional.one_hot(self.residue_codes.clamp_min(0), NUM_RESIDUE_CLASSES)
        counts = (one_hot * valid.unsqueeze(-1)).sum(dim=1).float()  # (d_hidden, NUM_RESIDUE_CLASSES)
        n_valid = valid.sum(dim=1)
        alive = n_valid >= self.top_k
        dominance_frac = counts.max(dim=1).values / n_valid.clamp_min(1).float()
        n_alive = alive.sum().clamp_min(1).float()

        def dominant(at: float) -> torch.Tensor:
            return (dominance_frac >= at - 1e-6) & alive  # the tolerance: 7/10 in float32 vs 0.7

        chance = 0.0
        if self.residue_counts.sum() > 0:  # nothing seen, nothing to compare against
            frequencies = self.residue_counts.float().cpu()
            generator = torch.Generator(device="cpu").manual_seed(0)
            draws = torch.multinomial(frequencies, 20_000 * self.top_k, replacement=True, generator=generator)
            chance_counts = torch.nn.functional.one_hot(draws.view(20_000, self.top_k), NUM_RESIDUE_CLASSES).sum(dim=1)
            chance = ((chance_counts.max(dim=1).values / self.top_k) >= threshold - 1e-6).float().mean().item()

        return {
            "collapse_rate": dominant(threshold).float().mean().item(),
            "collapse_rate_alive": (dominant(threshold).sum() / n_alive).item(),
            "collapse_rate_alive_t50": (dominant(0.5).sum() / n_alive).item(),
            "collapse_rate_alive_t90": (dominant(0.9).sum() / n_alive).item(),
            "chance_rate": chance,
            "n_alive": float(alive.sum().item()),
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
    cache_dir, remote_subpath = partition_root(
        config.cache_dir, config.remote_subpath, "discovery_val", config.partition_folders
    )
    reader = SafeTensorsReader(
        cache_dir=cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=remote_subpath,
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
