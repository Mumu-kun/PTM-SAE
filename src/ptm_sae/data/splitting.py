"""Homology-aware cluster partitioning using exact MILP (HiGHS) and iterative stratification."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import LinearConstraint, milp

from ptm_sae.data.schema import Protein, UnifiedResidueSite


def build_cluster_profiles(
    proteins: Sequence[Protein],
    sites: Sequence[UnifiedResidueSite],
    cluster_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """
    Aggregate protein token lengths and multi-label PTM counts up to cluster level.
    Returns cluster profiles dictionary and list of sorted observed PTM types.
    """
    cluster_map = cluster_mapping or {p.uniprot_id: p.uniprot_id for p in proteins}

    # Extract unique PTM types observed across all sites
    all_ptms: Set[str] = set()
    for s in sites:
        all_ptms.update(s.ptm_types)
    sorted_ptms = sorted(all_ptms)

    profiles: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
    cluster_profiles: Dict[str, Dict[str, int]],
    ptm_types: Sequence[str],
    target_discovery_ratio: float = 0.8,
    token_tolerance: float = 0.03,
) -> Dict[str, str]:
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
    tokens = np.array([cluster_profiles[c].get("tokens", 0) for c in cluster_ids], dtype=float)
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
    integrality = np.concatenate([np.ones(n_clusters), np.zeros(n_ptms)])  # x is binary, u is continuous
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
    lhs_all = np.concatenate([[lhs_token], np.full(n_ptms, -np.inf), np.full(n_ptms, -np.inf)])
    rhs_all = np.concatenate([[rhs_token], target_ptms, -target_ptms])
    constraints = LinearConstraint(A_all, lhs_all, rhs_all)

    # 4. Solve integer program via HiGHS
    res = milp(c=c_obj, integrality=integrality, bounds=(lb, ub), constraints=constraints)

    if res.success:
        x_sol = res.x[:n_clusters]
        return {
            c_id: ("discovery" if x_sol[i] > 0.5 else "held_out")
            for i, c_id in enumerate(cluster_ids)
        }

    # Fallback to greedy if hard token window is mathematically infeasible
    return solve_greedy_partition(cluster_profiles, ptm_types, target_discovery_ratio)


def solve_greedy_partition(
    cluster_profiles: Dict[str, Dict[str, int]],
    ptm_types: Sequence[str],
    target_discovery_ratio: float = 0.8,
) -> Dict[str, str]:
    """
    Greedy iterative stratification on cluster profiles.
    Prioritizes rarest PTMs first, assigning clusters to the partition with highest deficit.
    """
    cluster_ids = list(cluster_profiles.keys())
    if not cluster_ids:
        return {}

    target_held_out_ratio = 1.0 - target_discovery_ratio

    total_tokens = sum(cluster_profiles[c].get("tokens", 0) for c in cluster_ids)
    ptm_totals = {ptm: sum(cluster_profiles[c].get(ptm, 0) for c in cluster_ids) for ptm in ptm_types}

    target_disc_tokens = total_tokens * target_discovery_ratio
    target_held_tokens = total_tokens * target_held_out_ratio

    # Order PTMs from rarest to most frequent to steer rare sites first
    sorted_ptms = sorted(ptm_types, key=lambda p: ptm_totals[p])

    assigned: Dict[str, str] = {}
    current_tokens = {"discovery": 0, "held_out": 0}
    current_ptms = {
        "discovery": defaultdict(int),
        "held_out": defaultdict(int),
    }

    # Phase 1: Assign clusters carrying specific PTMs in order of rarity
    unassigned = set(cluster_ids)

    for ptm in sorted_ptms:
        if ptm_totals[ptm] == 0:
            continue

        candidates = [c for c in unassigned if cluster_profiles[c].get(ptm, 0) > 0]
        candidates.sort(key=lambda c: cluster_profiles[c].get(ptm, 0), reverse=True)

        for c in candidates:
            if c not in unassigned:
                continue

            target_disc_p = ptm_totals[ptm] * target_discovery_ratio
            target_held_p = ptm_totals[ptm] * target_held_out_ratio

            disc_deficit = (target_disc_p - current_ptms["discovery"][ptm]) / max(1.0, target_disc_p)
            held_deficit = (target_held_p - current_ptms["held_out"][ptm]) / max(1.0, target_held_p)

            # Assign to partition with higher unmet deficit
            choice = "discovery" if disc_deficit >= held_deficit else "held_out"

            # Check for token capacity overrun
            if choice == "discovery" and current_tokens["discovery"] > target_disc_tokens * 1.05:
                choice = "held_out"
            elif choice == "held_out" and current_tokens["held_out"] > target_held_tokens * 1.05:
                choice = "discovery"

            assigned[c] = choice
            unassigned.remove(c)
            current_tokens[choice] += cluster_profiles[c].get("tokens", 0)
            for p in ptm_types:
                current_ptms[choice][p] += cluster_profiles[c].get(p, 0)

    # Phase 2: Assign remaining unmodified clusters to balance total token volumes
    remaining = sorted(unassigned, key=lambda c: cluster_profiles[c].get("tokens", 0), reverse=True)
    for c in remaining:
        disc_deficit = target_disc_tokens - current_tokens["discovery"]
        held_deficit = target_held_tokens - current_tokens["held_out"]
        choice = "discovery" if disc_deficit >= held_deficit else "held_out"

        assigned[c] = choice
        current_tokens[choice] += cluster_profiles[c].get("tokens", 0)

    return assigned


def partition_dataset(
    proteins: Sequence[Protein],
    sites: Sequence[UnifiedResidueSite],
    cluster_mapping: Optional[Dict[str, str]] = None,
    target_discovery_ratio: float = 0.8,
) -> Tuple[List[Protein], Dict[str, str]]:
    """
    End-to-end homology cluster partitioning.
    Assigns each protein to 'discovery' or 'held_out' based on optimal cluster assignment.
    """
    profiles, ptm_types = build_cluster_profiles(proteins, sites, cluster_mapping)
    cluster_assignments = solve_milp_partition(
        cluster_profiles=profiles,
        ptm_types=ptm_types,
        target_discovery_ratio=target_discovery_ratio,
    )

    cluster_map = cluster_mapping or {p.uniprot_id: p.uniprot_id for p in proteins}
    sites_by_protein = defaultdict(list)
    for s in sites:
        sites_by_protein[s.uniprot_id].append(s)

    preprocessed: List[Protein] = []
    for p in proteins:
        c_id = cluster_map.get(p.uniprot_id, p.uniprot_id)
        partition = cluster_assignments.get(c_id, "discovery")
        p_sites = sites_by_protein[p.uniprot_id]

        stratum_counts = defaultdict(int)
        for s in p_sites:
            stratum_counts[s.stratum] += 1

        preprocessed.append(Protein(
            uniprot_id=p.uniprot_id,
            sequence=p.sequence,
            length=p.length,
            reviewed=p.reviewed,
            taxonomy_id=p.taxonomy_id,
            header=p.header,
            has_annotated_ptm=len(p_sites) > 0,
            stratum_counts=dict(stratum_counts),
            cluster_id=c_id,
            partition=partition,
        ))

    return preprocessed, cluster_assignments
