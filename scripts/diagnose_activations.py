"""Why is explained variance low at small k? Measures the geometry of the cached ESM-2 layer-24 activations.

Reads a sample of `discovery_train` tokens (a few shards, hydrated from the Hub on demand; nothing is
trained) and reports, without any SAE:

    scale      norm quantiles; how much variance a handful of dimensions carry (massive activations)
    spectrum   how many principal directions hold 50/80/90/95/99% of the variance; participation ratio
    identity   the share of variance that the amino acid alone explains; per-amino-acid spread
    position   whether chain ends carry outsized variance
    oracle     the explained variance an ideal k-sparse code reaches in the PCA basis and in the raw
               coordinate basis, per token (a reference an SAE's EV at the same k can be read against),
               and per amino acid at k = 32

Runs the same on a local PC and on Colab (public datasets: no token needed):

    uv run python scripts/diagnose_activations.py --max-tokens 300000
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch

from ptm_sae.extraction.reader import SafeTensorsReader
from ptm_sae.runtime import resolve_data_root
from ptm_sae.training.dataset import hydrate_corpus_file, partition_root
from ptm_sae.training.residue_dominance import _AA_TO_INDEX, _UNKNOWN_INDEX, NUM_RESIDUE_CLASSES

ACTIVATIONS_REPO = "mustafa-muhaimin/ptm-sae-dataset"
ACTIVATIONS_SUBPATH = "activations/esm2_t33_650M_UR50D/layer_24"
CORPUS_REPO = "mustafa-muhaimin/ptm-sae-corpus"
AMINO_ACIDS = [*sorted(_AA_TO_INDEX, key=_AA_TO_INDEX.get), "?"]
SPARSITY_LEVELS = (16, 32, 64, 128, 256, 512)
CHAIN_END = 5  # residues counted as "ends" at each terminus
CHUNK = 20_000  # tokens per matrix product / sort, bounds memory


def load_sample(partition: str, cache_dir: Path, corpus_dir: Path, max_tokens: int, n_shards: int, seed: int):
    """Activations (N, 1280) float32 plus per-token amino-acid code and distance to the nearest chain end.
    Tokens come from `n_shards` shards, an equal budget from each, so only those shards are downloaded."""
    corpus = hydrate_corpus_file("corpus.parquet", corpus_dir, CORPUS_REPO, "corpus")
    sequences = {
        row["uniprot_id"]: row["sequence"]
        for row in pq.read_table(corpus, columns=["uniprot_id", "sequence"], filters=[("partition", "==", partition)]).to_pylist()
    }
    root, remote = partition_root(cache_dir, ACTIVATIONS_SUBPATH, partition, True)
    reader = SafeTensorsReader(cache_dir=root, remote_repo_id=ACTIVATIONS_REPO, remote_subpath=remote)

    by_shard: dict[str, list[str]] = {}
    for uniprot_id, entry in reader.entries.items():
        if uniprot_id in sequences:
            by_shard.setdefault(entry["shard_file"], []).append(uniprot_id)
    rng = random.Random(seed)
    shards = rng.sample(sorted(by_shard), k=min(n_shards, len(by_shard)))
    per_shard = max_tokens // len(shards)

    activations, codes, edge_distance = [], [], []
    for shard in shards:
        proteins = by_shard[shard][:]
        rng.shuffle(proteins)
        taken = 0
        for uniprot_id in proteins:
            if taken >= per_shard:
                break
            x = reader.get_protein_activations(uniprot_id).float()
            length = min(x.shape[0], len(sequences[uniprot_id]))
            activations.append(x[:length])
            codes.append(torch.tensor([_AA_TO_INDEX.get(aa, _UNKNOWN_INDEX) for aa in sequences[uniprot_id][:length]]))
            positions = torch.arange(length)
            edge_distance.append(torch.minimum(positions, length - 1 - positions))
            taken += length
        print(f"  shard {shard}: {taken:,} tokens")
    reader.close()
    return torch.cat(activations), torch.cat(codes), torch.cat(edge_distance)


def share_table(counts_for: dict[str, float]) -> str:
    return "  ".join(f"{name}={value:.3f}" for name, value in counts_for.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", default="discovery_train", choices=["discovery_train", "discovery_val"])
    parser.add_argument("--max-tokens", type=int, default=300_000)
    parser.add_argument("--shards", type=int, default=4, help="shards to sample from: more shards, a more representative sample, more downloads")
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: <data root>/cache/activations/esm2_650m_l24")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="default: <data root>/data/processed")
    parser.add_argument("--out", type=Path, default=Path("runs/diagnosis/activation_diagnosis.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = resolve_data_root()
    cache_dir = args.cache_dir or root / "cache" / "activations" / "esm2_650m_l24"
    corpus_dir = args.corpus_dir or root / "data" / "processed"
    device = torch.device(args.device)
    report: dict = {"partition": args.partition, "device": str(device)}

    print(f"loading about {args.max_tokens:,} {args.partition} tokens...")
    x, codes, edge_distance = load_sample(args.partition, cache_dir, corpus_dir, args.max_tokens, args.shards, args.seed)
    n, d = x.shape
    report["tokens"] = n
    print(f"{n:,} tokens x {d} dims, {len(codes.unique())} residue classes")

    # Every float64 reduction below runs per chunk: a full float64 copy of 300k x 1280 tokens is 3 GB.
    x, codes, edge_distance = x.to(device), codes.to(device), edge_distance.to(device)
    mean = torch.stack([x[s : s + CHUNK].sum(0, dtype=torch.float64) for s in range(0, n, CHUNK)]).sum(0).div(n).float()
    centered = x - mean
    deviation = torch.cat([centered[s : s + CHUNK].pow(2).sum(1, dtype=torch.float64) for s in range(0, n, CHUNK)])  # squared distance to the mean, per token
    total_ss = deviation.sum().item()

    # scale: norms and how concentrated the variance is in a few dimensions
    norms = x.norm(dim=1)
    quantiles = torch.quantile(norms[:: max(1, n // 100_000)], torch.tensor([0.01, 0.5, 0.99], device=device))
    variance = torch.stack([centered[s : s + CHUNK].pow(2).sum(0, dtype=torch.float64) for s in range(0, n, CHUNK)]).sum(0) / n
    top_dims = variance.topk(5)
    report["scale"] = {
        "norm_p1_p50_p99_max": [*quantiles.tolist(), norms.max().item()],
        "top5_variance_dims": top_dims.indices.tolist(),
        "top5_variance_share": (top_dims.values.sum() / variance.sum()).item(),
        "largest_abs_mean_over_std": (mean.abs() / variance.sqrt().float().clamp_min(1e-12)).max().item(),
    }
    print("\nscale:", json.dumps(report["scale"]))

    # spectrum: eigen-decomposition of the covariance (accumulated in float64)
    cov = torch.zeros(d, d, dtype=torch.float64, device=device)
    for start in range(0, n, CHUNK):
        block = centered[start : start + CHUNK].double()
        cov += block.T @ block
    cov /= n
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues, eigenvectors = eigenvalues.flip(0).clamp_min(0), eigenvectors.flip(1)
    cumulative = eigenvalues.cumsum(0) / eigenvalues.sum()
    keep = torch.ones(d, dtype=torch.bool, device=device)
    keep[top_dims.indices] = False
    rest = torch.linalg.eigvalsh(cov[keep][:, keep]).clamp_min(0)
    report["spectrum"] = {
        "components_for_variance": {f"{q:.0%}": int((cumulative < q).sum().item()) + 1 for q in (0.5, 0.8, 0.9, 0.95, 0.99)},
        "top1_top10_share": [cumulative[0].item(), cumulative[9].item()],
        "participation_ratio": (eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum()).item(),
        "participation_ratio_without_top5_dims": (rest.sum() ** 2 / rest.pow(2).sum()).item(),
    }
    print("spectrum:", json.dumps(report["spectrum"]))

    # identity: how much does knowing only the amino acid explain, and how spread is each class
    class_counts = torch.bincount(codes, minlength=NUM_RESIDUE_CLASSES)
    class_means = torch.zeros(NUM_RESIDUE_CLASSES, d, device=device).index_add_(0, codes, x) / class_counts.clamp_min(1).unsqueeze(1)
    within = torch.cat([(x[s : s + CHUNK] - class_means[codes[s : s + CHUNK]]).pow(2).sum(1, dtype=torch.float64) for s in range(0, n, CHUNK)])
    within_by_class = torch.zeros(NUM_RESIDUE_CLASSES, dtype=torch.float64, device=device).index_add_(0, codes, within)
    report["identity"] = {
        "variance_explained_by_amino_acid_alone": 1 - within.sum().item() / total_ss,
        "per_amino_acid": {
            aa: {"tokens": int(class_counts[i]), "mean_norm": norms[codes == i].mean().item(), "share_of_within_variance": (within_by_class[i] / within.sum()).item()}
            for i, aa in enumerate(AMINO_ACIDS)
            if class_counts[i] > 0
        },
    }
    print(f"identity: amino acid alone explains {report['identity']['variance_explained_by_amino_acid_alone']:.1%} of the variance")

    # position: variance share of chain ends against their token share
    ends = edge_distance < CHAIN_END
    report["position"] = {"end_token_share": ends.float().mean().item(), "end_variance_share": (deviation[ends].sum() / total_ss).item(), "end_mean_norm": norms[ends].mean().item(), "inner_mean_norm": norms[~ends].mean().item()}
    print("position:", json.dumps(report["position"]))

    # oracle: best k-sparse code per token in an orthonormal basis keeps its k largest coefficients
    def kept_share(basis: torch.Tensor | None, k: int, tokens: torch.Tensor) -> torch.Tensor:
        """Squared norm kept per token by the k largest-magnitude coefficients (basis None: raw coordinates)."""
        kept = []
        for start in range(0, len(tokens), CHUNK):
            block = tokens[start : start + CHUNK]
            coefficients = block @ basis if basis is not None else block
            kept.append(coefficients.pow(2).topk(k, dim=1).values.sum(1).double())
        return torch.cat(kept)

    basis = eigenvectors.float()
    report["oracle"] = {"pca_dense_top_k_components": {}, "pca_sparse_per_token": {}, "raw_coordinate_sparse_per_token": {}}
    for k in SPARSITY_LEVELS:
        report["oracle"]["pca_dense_top_k_components"][k] = cumulative[k - 1].item()
        report["oracle"]["pca_sparse_per_token"][k] = kept_share(basis, k, centered).sum().item() / total_ss
        report["oracle"]["raw_coordinate_sparse_per_token"][k] = kept_share(None, k, centered).sum().item() / total_ss
    print("\nexplained variance of an ideal k-sparse code (no dictionary learning):")
    print(f"{'k':>5} {'dense PCA':>10} {'PCA sparse':>11} {'raw-coord sparse':>17}")
    for k in SPARSITY_LEVELS:
        row = [report["oracle"][name][k] for name in ("pca_dense_top_k_components", "pca_sparse_per_token", "raw_coordinate_sparse_per_token")]
        print(f"{k:>5} {row[0]:>10.1%} {row[1]:>11.1%} {row[2]:>17.1%}")

    # per amino acid at k = 32: where the unexplained variance sits
    k = 32
    kept = kept_share(basis, k, centered)
    per_class_total = torch.zeros(NUM_RESIDUE_CLASSES, dtype=torch.float64, device=device).index_add_(0, codes, deviation)
    per_class_kept = torch.zeros(NUM_RESIDUE_CLASSES, dtype=torch.float64, device=device).index_add_(0, codes, kept)
    report["pca_sparse_k32_by_amino_acid"] = {
        aa: {"explained_variance": (per_class_kept[i] / per_class_total[i]).item(), "share_of_unexplained": ((per_class_total[i] - per_class_kept[i]) / (total_ss - kept.sum())).item()}
        for i, aa in enumerate(AMINO_ACIDS)
        if class_counts[i] > 0
    }
    print(f"\nPCA-sparse EV at k={k} by amino acid, and each one's share of the unexplained variance:")
    print(share_table({aa: v["explained_variance"] for aa, v in report["pca_sparse_k32_by_amino_acid"].items()}))
    print(share_table({aa: v["share_of_unexplained"] for aa, v in report["pca_sparse_k32_by_amino_acid"].items()}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
