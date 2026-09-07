"""Homology-aware cluster partitioning using exact MILP (HiGHS) and iterative stratification."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from scipy.optimize import LinearConstraint, milp

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


def solve_milp_partition(
    cluster_profiles: dict[str, dict[str, int]],
    ptm_types: Sequence[str],
    target_discovery_ratio: float = 0.8,
    token_tolerance: float = 0.03,
) -> dict[str, str]:
    """
    Exact Mixed-Integer Linear Programming (MILP) cluster partitioner using HiGHS.
    Finds binary indicator x_i in {0, 1} minimizing weighted relative PTM deficit
    subject to a hard token quota window.
    """
    cluster_ids = list(cluster_profiles.keys())
    n_clusters = len(cluster_ids)
    n_ptms = len(ptm_types)

    if n_clusters == 0:
        return {}
    if n_clusters == 1:
        return {cluster_ids[0]: "discovery"}

    # 1. Target quotas and normalized penalty weights
    tokens = np.array(
        [cluster_profiles[c].get("tokens", 0) for c in cluster_ids], dtype=float
    )
    total_tokens = np.sum(tokens)
    target_tokens = total_tokens * target_discovery_ratio

    Y = np.zeros((n_clusters, n_ptms), dtype=float)
    for i, c in enumerate(cluster_ids):
        for j, ptm in enumerate(ptm_types):
            Y[i, j] = cluster_profiles[c].get(ptm, 0)

    total_ptms = np.sum(Y, axis=0)
    target_ptms = total_ptms * target_discovery_ratio

    # Penalize relative error so rare PTMs carry equal gravity to frequent ones
    weights = np.array([1.0 / max(1.0, t) for t in target_ptms])

    # 2. Decision variables: z = [x_0..x_{N-1}, u_0..u_{C-1}]
    c_obj = np.concatenate([np.zeros(n_clusters), weights])
    integrality = np.concatenate(
        [np.ones(n_clusters), np.zeros(n_ptms)]
    )  # x is binary, u is continuous
    lb = np.concatenate([np.zeros(n_clusters), np.zeros(n_ptms)])
    ub = np.concatenate([np.ones(n_clusters), np.full(n_ptms, np.inf)])

    # 3. Assemble linear constraints matrix
    # Hard token bounds: (1 - tol) * target <= sum(x_i * T_i) <= (1 + tol) * target
    A_token = np.concatenate([tokens, np.zeros(n_ptms)]).reshape(1, -1)
    lhs_token = target_tokens * (1.0 - token_tolerance)
    rhs_token = target_tokens * (1.0 + token_tolerance)

    # Absolute value bounds per PTM: Y*x - u <= Target and -Y*x - u <= -Target
    A_ptm_upper = np.zeros((n_ptms, n_clusters + n_ptms))
    A_ptm_lower = np.zeros((n_ptms, n_clusters + n_ptms))
    for j in range(n_ptms):
        A_ptm_upper[j, :n_clusters] = Y[:, j]
        A_ptm_upper[j, n_clusters + j] = -1.0

        A_ptm_lower[j, :n_clusters] = -Y[:, j]
        A_ptm_lower[j, n_clusters + j] = -1.0

    A_all = np.vstack([A_token, A_ptm_upper, A_ptm_lower])
    lhs_all = np.concatenate(
        [[lhs_token], np.full(n_ptms, -np.inf), np.full(n_ptms, -np.inf)]
    )
    rhs_all = np.concatenate([[rhs_token], target_ptms, -target_ptms])
    constraints = LinearConstraint(A_all, lhs_all, rhs_all)  # type: ignore[arg-type]

    # 4. Solve integer program via HiGHS with time limit and gap tolerance
    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=constraints,
        options={"time_limit": 30.0, "mip_rel_gap": 0.02},
    )

    if res.x is not None and len(res.x) >= n_clusters:
        x_sol = res.x[:n_clusters]
        assigned_tokens = np.sum(x_sol * tokens)
        if lhs_token <= assigned_tokens <= rhs_token:
            return {
                c_id: ("discovery" if x_sol[i] > 0.5 else "held_out")
                for i, c_id in enumerate(cluster_ids)
            }

    # Fallback to greedy if hard token window is mathematically infeasible or timed out
    return solve_greedy_partition(cluster_profiles, ptm_types, target_discovery_ratio)


def solve_greedy_partition(
    cluster_profiles: dict[str, dict[str, int]],
    ptm_types: Sequence[str],
    target_discovery_ratio: float = 0.8,
) -> dict[str, str]:
    """
    Greedy iterative stratification on cluster profiles.
    Prioritizes rarest PTMs first, assigning clusters to the partition with highest deficit.
    """
    cluster_ids = list(cluster_profiles.keys())
    if not cluster_ids:
        return {}

    target_held_out_ratio = 1.0 - target_discovery_ratio

    total_tokens = sum(cluster_profiles[c].get("tokens", 0) for c in cluster_ids)
    ptm_totals = {
        ptm: sum(cluster_profiles[c].get(ptm, 0) for c in cluster_ids)
        for ptm in ptm_types
    }

    target_disc_tokens = total_tokens * target_discovery_ratio
    target_held_tokens = total_tokens * target_held_out_ratio

    # Order PTMs from rarest to most frequent to steer rare sites first
    sorted_ptms = sorted(ptm_types, key=lambda p: ptm_totals[p])

    assigned: dict[str, str] = {}
    current_tokens = {"discovery": 0, "held_out": 0}
    current_ptms = {
        "discovery": defaultdict(int),
        "held_out": defaultdict(int),
    }

    # Phase 1: Assign clusters carrying specific PTMs in order of rarity
    unassigned_clusters = set(cluster_ids)

    for ptm in sorted_ptms:
        if ptm_totals[ptm] == 0:
            continue

        candidates = [
            cid for cid in unassigned_clusters if cluster_profiles[cid].get(ptm, 0) > 0
        ]
        candidates.sort(key=lambda cid: cluster_profiles[cid].get(ptm, 0), reverse=True)

        for cluster_id in candidates:
            if cluster_id not in unassigned_clusters:
                continue

            target_disc_p = ptm_totals[ptm] * target_discovery_ratio
            target_held_p = ptm_totals[ptm] * target_held_out_ratio

            discovery_deficit = (target_disc_p - current_ptms["discovery"][ptm]) / max(
                1.0, target_disc_p
            )
            held_out_deficit = (target_held_p - current_ptms["held_out"][ptm]) / max(
                1.0, target_held_p
            )

            # Assign to partition with higher unmet deficit
            choice = (
                "discovery" if discovery_deficit >= held_out_deficit else "held_out"
            )

            # Check for token capacity overrun
            if (
                choice == "discovery"
                and current_tokens["discovery"] > target_disc_tokens * 1.05
            ):
                choice = "held_out"
            elif (
                choice == "held_out"
                and current_tokens["held_out"] > target_held_tokens * 1.05
            ):
                choice = "discovery"

            assigned[cluster_id] = choice
            unassigned_clusters.remove(cluster_id)
            current_tokens[choice] += cluster_profiles[cluster_id].get("tokens", 0)
            for ptm_type in ptm_types:
                current_ptms[choice][ptm_type] += cluster_profiles[cluster_id].get(
                    ptm_type, 0
                )

    # Phase 2: Assign remaining unmodified clusters to balance total token volumes
    remaining_clusters = sorted(
        unassigned_clusters,
        key=lambda cid: cluster_profiles[cid].get("tokens", 0),
        reverse=True,
    )
    for cluster_id in remaining_clusters:
        discovery_deficit = target_disc_tokens - current_tokens["discovery"]
        held_out_deficit = target_held_tokens - current_tokens["held_out"]
        choice = "discovery" if discovery_deficit >= held_out_deficit else "held_out"

        assigned[cluster_id] = choice
        current_tokens[choice] += cluster_profiles[cluster_id].get("tokens", 0)

    return assigned


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


def partition_dataset(
    proteins: Sequence[Protein],
    sites: Sequence[UnifiedResidueSite],
    cluster_mapping: dict[str, str] | None = None,
    target_discovery_ratio: float = 0.8,
    subdivide_discovery: bool = True,
    train_val_ratio: float = 0.875,
) -> tuple[list[Protein], dict[str, str]]:
    """
    End-to-end homology cluster partitioning.
    Assigns each protein to 'discovery_train', 'discovery_val', or 'held_out'.
    """
    profiles, ptm_types = build_cluster_profiles(proteins, sites, cluster_mapping)
    cluster_assignments = solve_milp_partition(
        cluster_profiles=profiles,
        ptm_types=ptm_types,
        target_discovery_ratio=target_discovery_ratio,
    )

    # Optionally subdivide discovery into discovery_train (~70%) and discovery_val (~10%)
    if subdivide_discovery:
        discovery_cluster_ids = [
            cid for cid, part in cluster_assignments.items() if part == "discovery"
        ]
        discovery_sub_assignments = subdivide_discovery_clusters(
            cluster_profiles=profiles,
            discovery_cluster_ids=discovery_cluster_ids,
            train_ratio=train_val_ratio,
        )
        cluster_assignments.update(discovery_sub_assignments)

    cluster_map = cluster_mapping or {
        protein.uniprot_id: protein.uniprot_id for protein in proteins
    }
    sites_by_protein = defaultdict(list)
    for site in sites:
        cluster_id = cluster_map.get(site.uniprot_id, site.uniprot_id)
        site.partition = cluster_assignments.get(
            cluster_id,
            "discovery_train" if subdivide_discovery else "discovery",
        )
        sites_by_protein[site.uniprot_id].append(site)

    preprocessed_proteins: list[Protein] = []
    for protein in proteins:
        cluster_id = cluster_map.get(protein.uniprot_id, protein.uniprot_id)
        partition = cast(
            Literal["discovery", "discovery_train", "discovery_val", "held_out"],
            cluster_assignments.get(cluster_id, "discovery"),
        )
        protein_sites = sites_by_protein[protein.uniprot_id]

        stratum_counts = defaultdict(int)
        for site in protein_sites:
            stratum_counts[site.stratum] += 1

        preprocessed_proteins.append(
            Protein(
                uniprot_id=protein.uniprot_id,
                sequence=protein.sequence,
                length=protein.length,
                reviewed=protein.reviewed,
                taxonomy_id=protein.taxonomy_id,
                header=protein.header,
                has_annotated_ptm=len(protein_sites) > 0,
                stratum_counts=dict(stratum_counts),
                cluster_id=cluster_id,
                partition=partition,
            )
        )

    return preprocessed_proteins, cluster_assignments
