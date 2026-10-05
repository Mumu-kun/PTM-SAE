"""Homology-aware partition assignment: the joint three-way split of a built corpus into
discovery_train / discovery_val / held_out by whole homology clusters (`assign_partitions`), with the
cd-hit-2d cross-boundary audit that merges leaking clusters until the residual leak fraction is within
tolerance (`find_cross_partition_pairs`, `merge_leaking_clusters`).

The corpus build that used to live here (Swiss-Prot filtering, CD-HIT clustering at 40% identity, the
N1 two-way split, `run_m2`) was archived on the git branch `archive/corpus-build`; clusters now come from
the published corpus's `cluster_id` column.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ptm_sae.data.splitting import (
    PARTITIONS,
    SPLIT_VERSION,
    SplitSettings,
    balance_report,
    build_cluster_features,
    partition_clusters,
)

CDHIT_IDENTITY = 0.40


def check_cdhit_available() -> None:
    """cd-hit-2d is a system binary (not pip-installable) -- fail loudly and early rather than
    letting the audit's subprocess call produce an opaque FileNotFoundError."""
    if shutil.which("cd-hit-2d") is None:
        raise RuntimeError(
            "cd-hit-2d not found on PATH. Install cd-hit via your OS package manager (e.g. "
            "`sudo apt install cd-hit` on Debian/Ubuntu, `brew install cd-hit` on macOS, or build "
            "from https://github.com/weizhongli/cdhit), or skip the audit. See corpus/README.md."
        )


def _cdhit_word_size(identity: float) -> int:
    """CD-HIT's documented word-size bands: 5 for [0.7,1.0], 4 for [0.6,0.7), 3 for [0.5,0.6),
    2 for [0.4,0.5)."""
    for floor, word_size in ((0.7, 5), (0.6, 4), (0.5, 3)):
        if identity >= floor:
            return word_size
    return 2


# (reference partitions, query partition): cd-hit-2d reports every query sequence that reaches the
# identity threshold against any reference sequence -- i.e. a homology leak across that boundary.
QUERY_PARTITIONS = ("held_out", "discovery_val")  # the partitions audited against the rest

AUDIT_BOUNDARIES = (
    (("discovery_train", "discovery_val"), "held_out"),
    (("discovery_train",), "discovery_val"),
)


def find_cross_partition_pairs(
    proteins: pd.DataFrame,
    identity: float = CDHIT_IDENTITY,
    tmp_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Homology audit: runs cd-hit-2d across each partition boundary at `identity`.

    CD-HIT clusters compare members only to their cluster representative, so two proteins in
    DIFFERENT clusters can still exceed the threshold; this finds those. `proteins` needs
    uniprot_id, sequence and partition. Returns (reference_id, query_id) pairs."""
    check_cdhit_available()
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n = _cdhit_word_size(identity)
    usable = proteins[proteins["sequence"].str.len() >= n]  # shorter ones cannot form an n-mer
    pairs: list[tuple[str, str]] = []

    for index, (reference_parts, query_part) in enumerate(AUDIT_BOUNDARIES):
        fasta = {}
        for role, mask in (
            ("reference", usable["partition"].isin(reference_parts)),
            ("query", usable["partition"] == query_part),
        ):
            fasta[role] = tmp_dir / f"audit{index}_{role}.fasta"
            with open(fasta[role], "w") as f:
                f.writelines(
                    f">{row.uniprot_id}\n{row.sequence}\n" for row in usable[mask].itertuples()
                )

        out_path = tmp_dir / f"audit{index}_out"
        cmd = [
            "cd-hit-2d", "-i", str(fasta["reference"]), "-i2", str(fasta["query"]),
            "-o", str(out_path), "-c", str(identity), "-n", str(n),
            "-M", "0", "-T", "0", "-d", "0", "-l", str(max(1, n - 1)),
        ]  # fmt: skip
        print(f"[split] cd-hit-2d audit across the {query_part} boundary:")
        result = subprocess.run(cmd, text=True)  # noqa: S603 -- fixed local binary, never untrusted input
        if result.returncode != 0:
            raise RuntimeError(f"cd-hit-2d failed (rc={result.returncode})")

        reference_id = None
        with open(str(out_path) + ".clstr") as f:
            for line in f:
                if line.startswith(">Cluster"):
                    reference_id = None
                    continue
                member = re.search(r">(.+?)\.\.\.", line)
                if member is None:
                    continue
                if line.rstrip().endswith("*"):
                    reference_id = member.group(1)
                elif reference_id is not None:
                    pairs.append((reference_id, member.group(1)))
    return pairs


def merge_leaking_clusters(
    cluster_of: pd.Series, pairs: list[tuple[str, str]]
) -> tuple[pd.Series, int]:
    """Union-find over the clusters of every leaking (id, id) pair; the smallest cluster id
    represents each merged group. Returns the remapped `cluster_of` and how many clusters were
    absorbed."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        root_a, root_b = find(int(cluster_of[a])), find(int(cluster_of[b]))
        parent[max(root_a, root_b)] = min(root_a, root_b)

    root_of = {cid: find(cid) for cid in list(parent)}
    absorbed = sum(1 for cid, root in root_of.items() if cid != root)
    return cluster_of.map(lambda cid: root_of.get(int(cid), int(cid))), absorbed


def assign_partitions(
    corpus: pd.DataFrame,
    sites: pd.DataFrame,
    stratum_residues: dict[str, set[str]],
    settings: SplitSettings,
    identity: float = CDHIT_IDENTITY,
    tmp_dir: Path | None = None,
) -> tuple[pd.Series, dict[int, str], dict]:
    """Final 3-way partition of the finished corpus: one joint solve (data/splitting.py), then the
    cross-partition homology audit. Any pair that still reaches `identity` across a boundary
    merges its two clusters; a merged cluster takes the partition holding most of its tokens and
    only the local-search polish re-balances. The assignment is kept stable between rounds on
    purpose: re-solving from scratch reshuffles every cluster, which creates fresh boundary pairs
    faster than merging removes them (measured: 549 -> 475 -> 174 pairs, not converging). Up to
    `settings.audit_passes` merge rounds, or until every audited partition is within
    `settings.leak_target`; whatever remains is recorded (and gated by verify_outputs).

    `corpus`: uniprot_id, sequence, length, cluster_id. `sites`: uniprot_id, type_pooled,
    crosstalk. Returns (merged cluster id per uniprot_id, partition per cluster id, report)."""
    cluster_of = corpus.set_index("uniprot_id")["cluster_id"]
    audit: list[dict] = []
    previous_code = None  # partition code per protein from the previous round

    for attempt in range(settings.audit_passes + 1):
        frame = corpus.assign(cluster_id=corpus["uniprot_id"].map(cluster_of))
        site_frame = sites.assign(cluster_id=sites["uniprot_id"].map(cluster_of))
        features = build_cluster_features(frame, site_frame, stratum_residues, settings)

        start = None
        if previous_code is not None:
            token_by_code = (
                frame.assign(code=frame["uniprot_id"].map(previous_code))
                .groupby(["cluster_id", "code"])["length"]
                .sum()
                .unstack(fill_value=0)
                .reindex(index=features.cluster_ids, columns=range(len(PARTITIONS)), fill_value=0)
            )
            start = token_by_code.to_numpy().argmax(axis=1)
        codes = partition_clusters(features, settings, start=start)

        partition_of_cluster = {
            int(cid): PARTITIONS[code]
            for cid, code in zip(features.cluster_ids, codes, strict=True)
        }
        previous_code = frame["cluster_id"].map(dict(zip(features.cluster_ids.tolist(), codes.tolist(), strict=True))).set_axis(frame["uniprot_id"])
        if not settings.homology_audit:
            audit.append({"pass": attempt, "skipped": True})
            break

        audited = frame.assign(partition=frame["cluster_id"].map(partition_of_cluster))
        pairs = find_cross_partition_pairs(audited, identity, tmp_dir)
        partition_of_protein = audited.set_index("uniprot_id")["partition"]
        leaking_queries = {query for _, query in pairs}
        audit.append(
            {
                "pass": attempt,
                "cross_partition_pairs": len(pairs),
                # Proteins of the smaller partition with a homolog across the boundary, and how many
                # were audited: the fraction is what verify_outputs gates on.
                "queries": {name: int((partition_of_protein == name).sum()) for name in QUERY_PARTITIONS},
                "leaks": {
                    name: int((partition_of_protein.reindex(list(leaking_queries)) == name).sum())
                    for name in QUERY_PARTITIONS
                },
            }
        )
        within_target = all(
            audit[-1]["leaks"][name] <= settings.leak_target * audit[-1]["queries"][name]
            for name in QUERY_PARTITIONS
        )
        if not pairs or within_target or attempt == settings.audit_passes:
            break

        cluster_of, n_merged = merge_leaking_clusters(cluster_of, pairs)
        audit[-1]["clusters_merged"] = n_merged

    report = {
        "split_version": SPLIT_VERSION,
        "balance": balance_report(features, codes, settings),
        "homology_audit": audit,
    }
    return cluster_of, partition_of_cluster, report
