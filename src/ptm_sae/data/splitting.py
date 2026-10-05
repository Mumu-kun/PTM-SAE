"""Joint cluster-level partitioning into discovery_train / discovery_val / held_out.

The homology cluster is the atomic unit: every protein in a cluster lands in the same partition.
Within that constraint the three partitions are made statistically alike, so a validation loss or a
held-out re-estimate describes the same population the SAE trains on. "Alike" is measured on:
tokens, sites per PTM type, residues per chemical stratum (the enrichment denominators), tokens per
cluster-size bin (so val is not just singletons), proteins carrying a site, and crosstalk sites.

Method (tuned on the real corpus, see docs/thesis-meeting-split-strategy.md):
1. A MILP (HiGHS) decides the few hundred largest clusters exactly, with the long tail relaxed.
   Alone that is degenerate (fractional tails make every target trivially reachable), but it places
   the lumps that dominate the cluster-size features.
2. The tail is rounded deterministically by token-weighted error diffusion.
3. A deterministic local search moves single clusters between partitions while the weighted
   relative deviation falls, inside the token windows and per-type site floors.
4. The result is verified: token windows and floors must hold, or the split fails loudly.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

PARTITIONS = ("discovery_train", "discovery_val", "held_out")
SPLIT_VERSION = "milp-polish-v1"

# (feature name, min cluster size, max cluster size): tokens of clusters in that size band.
SIZE_BINS = (
    ("size_1", 1, 1),
    ("size_2_3", 2, 3),
    ("size_4_10", 4, 10),
    ("size_gt_10", 11, 10**9),
)
OTHER_TYPE_MARKER = "::other"  # pooled rare-type classes (e.g. "K::other_PTM") are not balanced
FLOOR_SLACK = 0.9  # a floor never exceeds 90% of the type's target share, so it stays feasible
SOLVER_FLOOR_MARGIN = 1.05  # the MILP aims above each floor so rounding cannot dip below it
FLOOR_PENALTY = 10.0  # local-search penalty per unit of relative floor shortfall


@dataclass(frozen=True)
class SplitSettings:
    fractions: tuple[float, float, float] = (0.70, 0.10, 0.20)  # token share: train, val, held
    token_tolerance: float = 0.02  # hard window around each val/held token target
    binary_clusters: int = 150  # largest clusters solved as integers; the tail is relaxed
    node_limit: int = 400  # branch-and-bound node cap: makes the solve deterministic
    time_limit_s: float = 300.0
    mip_rel_gap: float = 0.02
    min_type_sites: int = 300  # PTM types below this are too sparse to balance
    max_cluster_share: float = 0.10  # a type with >=10% of its sites in one cluster is lumpy
    held_min_sites: int = 200  # V12b's per-type floor in held_out
    val_min_sites: int = 100
    polish_sweeps: int = 60
    audit_passes: int = 8  # cross-partition homology audit + cluster-merge rounds
    leak_target: float = 0.005  # stop merging once every audited partition is at or below this leak fraction
    homology_audit: bool = True  # False only where cd-hit-2d is unavailable (recorded in the manifest)


@dataclass
class ClusterFeatures:
    cluster_ids: np.ndarray  # (N,) sorted
    tokens: np.ndarray  # (N,)
    matrix: np.ndarray  # (F, N) per-cluster contribution to each balance feature
    names: list[str]
    type_rows: list[int]  # rows of `matrix` that are per-type site counts (floors apply)
    excluded_types: dict[str, str]


def build_cluster_features(
    proteins: pd.DataFrame,
    sites: pd.DataFrame,
    stratum_residues: dict[str, set[str]],
    settings: SplitSettings,
) -> ClusterFeatures:
    """`proteins`: uniprot_id, cluster_id, sequence, length. `sites`: uniprot_id, cluster_id,
    type_pooled, crosstalk (one row per labelled site)."""
    cluster_ids = np.sort(proteins["cluster_id"].unique())
    n = len(cluster_ids)
    index = pd.Index(cluster_ids)
    p_row = index.get_indexer(proteins["cluster_id"])
    s_row = index.get_indexer(sites["cluster_id"])

    tokens = np.bincount(p_row, weights=proteins["length"].to_numpy(float), minlength=n)
    cluster_size = np.bincount(p_row, minlength=n)
    features: dict[str, np.ndarray] = {}

    # 1. Homology structure: tokens per cluster-size band
    for name, lo, hi in SIZE_BINS:
        features[name] = np.where((cluster_size >= lo) & (cluster_size <= hi), tokens, 0.0)

    # 2. Enrichment denominators: residues per chemical stratum
    for stratum, residues in stratum_residues.items():
        per_protein = sum(proteins["sequence"].str.count(r) for r in residues)
        features[f"res_{stratum}"] = np.bincount(
            p_row, weights=per_protein.to_numpy(float), minlength=n
        )

    # 3. Labels: sites per PTM type, skipping types too sparse or concentrated to balance
    excluded: dict[str, str] = {}
    type_totals = sites["type_pooled"].value_counts()
    for ptm_type, total in type_totals.items():
        if OTHER_TYPE_MARKER in ptm_type:
            excluded[ptm_type] = "pooled rare-type class"
            continue
        if total < settings.min_type_sites:
            excluded[ptm_type] = f"fewer than {settings.min_type_sites} sites"
            continue
        per_cluster = np.bincount(
            s_row[(sites["type_pooled"] == ptm_type).to_numpy()], minlength=n
        ).astype(float)
        if per_cluster.max() >= settings.max_cluster_share * total:
            excluded[ptm_type] = "one cluster holds too large a share of its sites"
            continue
        features[f"type:{ptm_type}"] = per_cluster

    features["sites_all"] = np.bincount(s_row, minlength=n).astype(float)
    has_site = proteins["uniprot_id"].isin(sites["uniprot_id"]).to_numpy(float)
    features["proteins_with_site"] = np.bincount(p_row, weights=has_site, minlength=n)
    features["crosstalk_sites"] = np.bincount(
        s_row, weights=sites["crosstalk"].astype(bool).to_numpy(float), minlength=n
    )

    names = list(features)
    return ClusterFeatures(
        cluster_ids=cluster_ids,
        tokens=tokens,
        matrix=np.array([features[name] for name in names]),
        names=names,
        type_rows=[i for i, name in enumerate(names) if name.startswith("type:")],
        excluded_types=excluded,
    )


def _floors(features: ClusterFeatures, settings: SplitSettings) -> np.ndarray:
    """(3, F) minimum sites per partition for each per-type feature (0 elsewhere)."""
    floors = np.zeros((3, len(features.names)))
    totals = features.matrix.sum(axis=1)
    for row in features.type_rows:
        floors[1, row] = np.floor(min(settings.val_min_sites, FLOOR_SLACK * settings.fractions[1] * totals[row]))
        floors[2, row] = np.floor(min(settings.held_min_sites, FLOOR_SLACK * settings.fractions[2] * totals[row]))
    return floors


def _solve_relaxed(features: ClusterFeatures, settings: SplitSettings) -> np.ndarray:
    """MILP over the largest clusters with the tail relaxed, then deterministic rounding."""
    tokens, matrix = features.tokens, features.matrix
    n, n_feat = len(tokens), matrix.shape[0]
    totals = matrix.sum(axis=1)
    fractions = np.array(settings.fractions)
    floors = _floors(features, settings)
    n_vars = 2 * n + 2 * n_feat  # y_val (n), y_held (n), then |deviation| slacks u_val, u_held

    # 1. MILP: rows = one partition per cluster, token windows, |deviation| bounds, type floors
    y_sparse = sp.csr_matrix(matrix)

    def block(offset: int, values: sp.csr_matrix) -> sp.csr_matrix:
        rows = values.shape[0]
        return sp.hstack(
            [sp.csr_matrix((rows, offset)), values, sp.csr_matrix((rows, n_vars - offset - n))]
        ).tocsr()

    blocks = [sp.hstack([sp.identity(n), sp.identity(n), sp.csr_matrix((n, 2 * n_feat))]).tocsr()]
    row_lo, row_hi = [-np.inf] * n, [1.0] * n
    token_row = sp.csr_matrix(tokens.reshape(1, -1))
    for k, (y_offset, u_offset) in ((1, (0, 2 * n)), (2, (n, 2 * n + n_feat))):
        target = fractions[k] * totals
        slack = sp.csr_matrix(
            (-np.ones(n_feat), (np.arange(n_feat), u_offset + np.arange(n_feat))),
            shape=(n_feat, n_vars),
        )
        blocks += [block(y_offset, token_row)]
        row_lo.append(fractions[k] * tokens.sum() * (1 - settings.token_tolerance))
        row_hi.append(fractions[k] * tokens.sum() * (1 + settings.token_tolerance))
        blocks += [block(y_offset, y_sparse) + slack, block(y_offset, -y_sparse) + slack]
        row_lo += [-np.inf] * (2 * n_feat)
        row_hi += list(target) + list(-target)
        floor_rows = np.flatnonzero(floors[k])
        blocks.append(block(y_offset, y_sparse[floor_rows]))
        row_lo += list(np.ceil(floors[k][floor_rows] * SOLVER_FLOOR_MARGIN))
        row_hi += [np.inf] * len(floor_rows)

    weights = np.concatenate(
        [
            np.zeros(2 * n),
            1 / np.maximum(fractions[1] * totals, 1),
            1 / np.maximum(fractions[2] * totals, 1),
        ]
    )
    # Integer variables: the clusters contributing most (relatively) to any balance feature
    lumpiness = (matrix / np.maximum(totals[:, None], 1)).max(axis=0)
    largest = np.argsort(-lumpiness, kind="stable")[: settings.binary_clusters]
    integrality = np.zeros(n_vars)
    integrality[largest] = integrality[n + largest] = 1
    upper = np.concatenate([np.ones(2 * n), np.full(2 * n_feat, np.inf)])

    result = milp(
        c=weights,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), upper),
        constraints=LinearConstraint(sp.vstack(blocks).tocsr(), row_lo, row_hi),
        options={
            "time_limit": settings.time_limit_s,
            "node_limit": settings.node_limit,
            "mip_rel_gap": settings.mip_rel_gap,
            "disp": False,
        },
    )
    if result.x is None:
        raise RuntimeError(
            f"Split MILP found no feasible assignment (status {result.status}: {result.message}). "
            "Loosen token_tolerance or the type floors."
        )
    p_val, p_held = result.x[:n], result.x[n : 2 * n]

    # 2. Round: integers directly, the relaxed tail by token-weighted error diffusion
    codes = np.zeros(n, int)
    codes[largest] = np.where(p_held[largest] > 0.5, 2, np.where(p_val[largest] > 0.5, 1, 0))
    tail = np.setdiff1d(np.arange(n), largest)
    tail = tail[np.argsort(-tokens[tail], kind="stable")]
    owed = np.zeros(3)
    for i in tail:
        owed += tokens[i] * np.array([max(0.0, 1 - p_val[i] - p_held[i]), p_val[i], p_held[i]])
        codes[i] = int(np.argmax(owed))
        owed[codes[i]] -= tokens[i]
    return codes


def polish_partition(
    features: ClusterFeatures, start: np.ndarray, settings: SplitSettings
) -> np.ndarray:
    """Deterministic local search from `start`: single-cluster moves that lower the weighted
    deviation, never worsening a token-window violation (a feasible start stays feasible; an
    infeasible one, e.g. after clusters were merged, is steered back inside), then verification
    that the hard constraints hold. Raises if they cannot."""
    tokens, matrix = features.tokens, features.matrix
    totals = matrix.sum(axis=1)
    fractions = np.array(settings.fractions)
    floors = _floors(features, settings)
    codes = start.copy()

    target = fractions[:, None] * totals[None, :]
    inverse_target = 1 / np.maximum(target, 1)
    inverse_target[0] = 0  # train's deviation is implied by val's and held's
    window_lo = fractions * tokens.sum() * (1 - settings.token_tolerance)
    window_hi = fractions * tokens.sum() * (1 + settings.token_tolerance)
    window_lo[0], window_hi[0] = 0, np.inf
    sums = np.array([matrix[:, codes == k].sum(axis=1) for k in range(3)])
    part_tokens = np.array([tokens[codes == k].sum() for k in range(3)])

    def violation(k: int, token_count: float) -> float:
        outside = max(0.0, window_lo[k] - token_count, token_count - window_hi[k])
        return outside / (fractions[k] * tokens.sum())

    def cost(k: int, vector: np.ndarray, token_count: float) -> float:
        deviation = float(inverse_target[k] @ np.abs(vector - target[k]))
        shortfall = np.maximum(floors[k] - vector, 0) / np.maximum(floors[k], 1)
        return deviation + FLOOR_PENALTY * float(shortfall.sum() + violation(k, token_count))

    for _ in range(settings.polish_sweeps):
        moved = 0
        for i in np.argsort(-tokens, kind="stable"):
            src, column = codes[i], matrix[:, i]
            src_tokens = part_tokens[src] - tokens[i]
            if violation(src, src_tokens) > violation(src, part_tokens[src]):
                continue
            src_gain = cost(src, sums[src], part_tokens[src]) - cost(src, sums[src] - column, src_tokens)
            best, best_gain = src, 1e-12
            for dst in range(3):
                if dst == src:
                    continue
                dst_tokens = part_tokens[dst] + tokens[i]
                if violation(dst, dst_tokens) > violation(dst, part_tokens[dst]):
                    continue
                gain = src_gain - (
                    cost(dst, sums[dst] + column, dst_tokens) - cost(dst, sums[dst], part_tokens[dst])
                )
                if gain > best_gain:
                    best, best_gain = dst, gain
            if best == src:
                continue
            sums[src] -= column
            sums[best] += column
            part_tokens[src] -= tokens[i]
            part_tokens[best] += tokens[i]
            codes[i] = best
            moved += 1
        if moved == 0:
            break

    # Verify: the hard constraints must hold on the final integer assignment
    if not np.all((window_lo <= part_tokens) & (part_tokens <= window_hi)):
        raise ValueError(f"Token windows violated after rounding: {part_tokens / tokens.sum()}")
    short = [
        features.names[j]
        for k in (1, 2)
        for j in np.flatnonzero(sums[k] < floors[k])
    ]
    if short:
        raise ValueError(f"Per-type site floors not met after rounding: {sorted(set(short))}")
    return codes


def partition_clusters(
    features: ClusterFeatures, settings: SplitSettings, start: np.ndarray | None = None
) -> np.ndarray:
    """Returns the partition code (index into PARTITIONS) of every cluster. Without `start` this
    is the full solve (MILP + rounding + polish); with `start` (e.g. the previous assignment after
    clusters were merged) only the polish runs, so the assignment stays stable."""
    codes = _solve_relaxed(features, settings) if start is None else start
    return polish_partition(features, codes, settings)


def balance_report(
    features: ClusterFeatures, codes: np.ndarray, settings: SplitSettings
) -> dict:
    """Achieved vs target share per feature, plus summary statistics, for split_manifest.json."""
    totals = features.matrix.sum(axis=1)
    fractions = settings.fractions
    shares = np.array(
        [features.matrix[:, codes == k].sum(axis=1) / np.maximum(totals, 1) for k in range(3)]
    )
    deviation = np.abs(shares[1:] - np.array(fractions[1:])[:, None]) / np.array(fractions[1:])[:, None]
    floors = _floors(features, settings)
    return {
        "token_share": {
            PARTITIONS[k]: float(features.tokens[codes == k].sum() / features.tokens.sum())
            for k in range(3)
        },
        "features": {
            name: {
                "total": float(totals[j]),
                "share": {PARTITIONS[k]: round(float(shares[k, j]), 5) for k in range(3)},
                "relative_deviation": {
                    PARTITIONS[k]: round(float(deviation[k - 1, j]), 5) for k in (1, 2)
                },
            }
            for j, name in enumerate(features.names)
        },
        "summary": {
            PARTITIONS[k]: {
                "mean_relative_deviation": float(deviation[k - 1].mean()),
                "max_relative_deviation": float(deviation[k - 1].max()),
                "features_over_25pct": int((deviation[k - 1] > 0.25).sum()),
            }
            for k in (1, 2)
        },
        "type_floors": {
            features.names[j][5:]: {
                PARTITIONS[k]: [
                    int(features.matrix[j, codes == k].sum()),
                    int(floors[k, j]),
                ]
                for k in (1, 2)
            }
            for j in features.type_rows
        },
        "excluded_types": features.excluded_types,
        "settings": asdict(settings),
        "split_version": SPLIT_VERSION,
    }
