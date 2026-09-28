"""Cluster-profile aggregation and discovery_train/discovery_val subdivision.

The exact-MILP/greedy 2-way discovery/held_out partitioner that used to live here was retired
once N1's already-computed CD-HIT clustering + split became the corpus's canonical partition
source (see the implementation plan, Stage 2, decision 7) -- corpus-building no longer re-derives
a split in-process. `build_cluster_profiles`/`subdivide_discovery_clusters` remain: they're reused
to subdivide N1's `discovery` clusters into `discovery_train`/`discovery_val`.
"""

from collections import defaultdict
from collections.abc import Sequence

from ptm_sae.data.schema import Protein, UnifiedResidueSite


def build_cluster_profiles(
    proteins: Sequence[Protein],
    sites: Sequence[UnifiedResidueSite],
    cluster_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """
    Aggregate protein token lengths and multi-label PTM counts up to cluster level.
    Returns cluster profiles dictionary and list of sorted observed PTM types.
    """
    cluster_map = cluster_mapping or {p.uniprot_id: p.uniprot_id for p in proteins}

    # Extract unique PTM types observed across all sites
    all_ptms: set[str] = set()
    for s in sites:
        all_ptms.update(s.ptm_types)
    sorted_ptms = sorted(all_ptms)

    profiles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Sum sequence tokens per homology cluster
    for p in proteins:
        c_id = cluster_map.get(p.uniprot_id, p.uniprot_id)
        profiles[c_id]["tokens"] += p.length

    # Count distinct PTM occurrences per homology cluster
    for s in sites:
        c_id = cluster_map.get(s.uniprot_id, s.uniprot_id)
        for ptm in s.ptm_types:
            profiles[c_id][ptm] += 1

    return dict(profiles), sorted_ptms


def subdivide_discovery_clusters(
    cluster_profiles: dict[str, dict[str, int]],
    discovery_cluster_ids: Sequence[str],
    train_ratio: float = 0.875,
) -> dict[str, str]:
    """
    Subdivides discovery clusters into discovery_train and discovery_val
    preserving atomic cluster blocks. Default 0.875 ratio splits an 80% discovery set
    into ~70% train and ~10% val overall.
    """
    if not discovery_cluster_ids:
        return {}

    total_tokens = sum(
        cluster_profiles[cid].get("tokens", 0) for cid in discovery_cluster_ids
    )
    target_train_tokens = total_tokens * train_ratio

    # Sort deterministically by token size descending for tight knapsack binning
    sorted_clusters = sorted(
        discovery_cluster_ids,
        key=lambda cid: (cluster_profiles[cid].get("tokens", 0), cid),
        reverse=True,
    )

    discovery_sub_assignments: dict[str, str] = {}
    accumulated_train_tokens = 0

    for cluster_id in sorted_clusters:
        cluster_tokens = cluster_profiles[cluster_id].get("tokens", 0)
        if (
            accumulated_train_tokens + cluster_tokens <= target_train_tokens
        ) or not discovery_sub_assignments:
            discovery_sub_assignments[cluster_id] = "discovery_train"
            accumulated_train_tokens += cluster_tokens
        else:
            discovery_sub_assignments[cluster_id] = "discovery_val"

    return discovery_sub_assignments
