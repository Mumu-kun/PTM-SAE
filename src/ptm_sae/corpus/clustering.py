"""M2 -- Corpus construction. Ported from N1.ipynb cells 1 (cd-hit check) and 11-12.

Inputs: M1 Swiss-Prot (+ a lightweight, unfiltered read of M1's raw PTM sources).
Emits: corpus.parquet (accession, sequence, length, cluster_id, split); stratum_counts.json;
       split_manifest.json.
Gate: protein count within 15,000-20,000; each primary type >=1000 sites in the discovery
      partition (else halve holdout_fraction to 0.15 and re-split).

ORDERING NOTE: M3's split-inheritance step requires cluster->split assignment to already exist,
but a stratified split needs to know which proteins carry which PTM types before M3's full
filtering cascade has run. Resolved by doing a LIGHTWEIGHT, UNFILTERED raw positional join here
(no evidence filtering, no dedup -- just "does this accession have >=1 raw reported site of type
X") purely to drive cluster stratification; M3 then re-derives the real, filtered labels
independently and inherits the split column by cluster.

CD-HIT identity is hardcoded at 0.40 (the user's decision, not a tunable) -- see
`CorpusPaths`/`Config.cdhit_identity` for where that value lives; `run_cdhit`'s own `identity`
parameter exists only because CD-HIT's word-size band logic depends on it, never to expose a
different threshold to callers here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ptm_sae.corpus.config import (
    CorpusPaths,
    config_fingerprint,
    find_cached,
    normalize_ptm_type,
    sha256_file,
)

CDHIT_IDENTITY = 0.40


def check_cdhit_available() -> None:
    """cd-hit is a system binary (not pip-installable) -- fail loudly and early rather than
    letting `run_cdhit`'s subprocess call produce an opaque FileNotFoundError."""
    if shutil.which("cd-hit") is None or shutil.which("cd-hit-2d") is None:
        raise RuntimeError(
            "cd-hit / cd-hit-2d not found on PATH. Install via your OS package manager before "
            "running M2 (e.g. `sudo apt install cd-hit` on Debian/Ubuntu, `brew install cd-hit` "
            "on macOS, or build from https://github.com/weizhongli/cdhit). See corpus/README.md."
        )


def filter_swissprot(swissprot_tsv: Path, max_length: int) -> pd.DataFrame:
    """M2 step 1: reviewed human, length <= max_length. Re-asserts the filter even though M1's
    UniProt query already applies it, so this function is correct fed any TSV."""
    df = pd.read_csv(swissprot_tsv, sep="\t")
    col_map = {
        "accession": next(
            c for c in df.columns if c.lower() in ("entry", "accession", "entry name")
        ),
        "length": next(c for c in df.columns if c.lower() == "length"),
        "sequence": next(c for c in df.columns if c.lower() == "sequence"),
    }
    out = df.rename(columns={v: k for k, v in col_map.items()})[
        ["accession", "length", "sequence"]
    ].copy()
    out = out[out["length"] <= max_length].reset_index(drop=True)
    out["sequence"] = out["sequence"].astype(str).str.upper()
    return out


def compute_stratum_counts(
    corpus: pd.DataFrame, stratum_residues: dict[str, set]
) -> dict[str, int]:
    """M2 steps 2-3: exact residue counts per stratum + total (must be exact, not sampled -- every
    downstream enrichment value depends on this background denominator being exact)."""
    from tqdm.auto import tqdm

    counts = {k: 0 for k in stratum_residues}
    total = 0
    for seq in tqdm(
        corpus["sequence"], desc="[M2] Stratum counts", unit="protein", leave=False
    ):
        total += len(seq)
        for stratum, residues in stratum_residues.items():
            counts[stratum] += sum(seq.count(r) for r in residues)
    counts["total_residues"] = total
    counts["n_proteins"] = len(corpus)
    return counts


def run_cdhit(
    corpus: pd.DataFrame, identity: float = CDHIT_IDENTITY, tmp_dir: Path | None = None
) -> dict[str, int]:
    """M2 step 4: CD-HIT clustering at `identity`. Requires the `cd-hit` binary on PATH. Word size
    follows CD-HIT's documented threshold bands: n=5 for [0.7,1.0], n=4 for [0.6,0.7),
    n=3 for [0.5,0.6), n=2 for [0.4,0.5). Returns {accession: cluster_id}.

    Two real behaviours handled explicitly, both silent otherwise:
    (1) CD-HIT's own "-l" (throw-away length) defaults to 10 -- any sequence <=10 residues is
        silently dropped. Fixed by setting -l to n-1, the lowest CD-HIT will accept for the
        chosen word length.
    (2) A sequence shorter than word length n can never form an n-mer at that word length. Fixed
        by routing sequences shorter than n around CD-HIT entirely and assigning each its own
        singleton cluster."""
    check_cdhit_available()
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if identity >= 0.7:
        n = 5
    elif identity >= 0.6:
        n = 4
    elif identity >= 0.5:
        n = 3
    else:
        n = 2

    seq_lens = corpus["sequence"].str.len()
    too_short_mask = seq_lens < n
    cdhit_corpus = corpus.loc[~too_short_mask]
    too_short = corpus.loc[too_short_mask]

    fasta_path = tmp_dir / "corpus.fasta"
    out_path = tmp_dir / "corpus_cdhit"
    with open(fasta_path, "w") as f:
        f.writelines(
            f">{row['accession']}\n{row['sequence']}\n"
            for _, row in cdhit_corpus.iterrows()
        )

    min_len_kept = max(1, n - 1)
    cmd = [
        "cd-hit",
        "-i",
        str(fasta_path),
        "-o",
        str(out_path),
        "-c",
        str(identity),
        "-n",
        str(n),
        "-M",
        "0",
        "-T",
        "0",
        "-d",
        "0",
        "-l",
        str(min_len_kept),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 -- cd-hit, a fixed local binary, never untrusted input
    if result.returncode != 0:
        raise RuntimeError(
            f"cd-hit failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    clstr_path = Path(str(out_path) + ".clstr")
    assignment: dict[str, int] = {}
    current_cluster = -1
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = int(line.strip().split()[-1])
            else:
                acc = line.split(">")[1].split("...")[0]
                assignment[acc] = current_cluster

    next_cluster = (max(assignment.values()) + 1) if assignment else 0
    for _, row in too_short.iterrows():
        assignment[row["accession"]] = next_cluster
        next_cluster += 1

    return assignment


def raw_type_hits_by_accession(
    m1_raw_paths: dict[str, Path],
    parsers: dict[str, Callable],
    cfg,
    corpus_seq: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Lightweight unfiltered raw site counts per accession per canonical type, used only to
    drive M2's stratified split (see module docstring). Keeps EVERY canonical type, not just
    `cfg.primary_types`, and resolves types residue-aware where the corpus sequence is
    available."""
    known = set(cfg.ptm_type_to_stratum)
    hits: dict[str, dict[str, int]] = {}
    for label, path in m1_raw_paths.items():
        if label not in parsers:
            continue
        df = parsers[label](path)
        if "type" not in df.columns or not len(df):
            continue
        df = df.copy()
        if corpus_seq is not None and "position" in df.columns:
            residues = []
            for acc, pos in zip(
                df["accession"].astype(str), df["position"], strict=False
            ):
                seq = corpus_seq.get(acc)
                if seq is None or pd.isna(pos):
                    residues.append(None)
                    continue
                p = int(pos)
                residues.append(seq[p - 1] if 1 <= p <= len(seq) else None)
            df["type"] = [
                normalize_ptm_type(t, known_types=known, residue=r)
                for t, r in zip(df["type"], residues, strict=False)
            ]
        else:
            df["type"] = df["type"].map(
                lambda t: normalize_ptm_type(t, known_types=known)
            )
        df = df[df["type"].isin(known)]
        for acc, sub in df.groupby("accession"):
            d = hits.setdefault(acc, {})
            for t, n in sub["type"].value_counts().items():
                d[t] = d.get(t, 0) + int(n)
    return hits


def iterative_stratified_split(
    cluster_type_counts: dict[int, dict[str, int]],
    all_cluster_ids: list[int],
    holdout_fraction: float,
    seed: int,
) -> dict[int, str]:
    """Greedy multi-label iterative stratification (Sechidis et al. 2011, simplified) at cluster
    level. Clusters with no primary-type sites (most proteins) are split by plain ratio at the
    end, since they only affect background sizing, not per-type balance."""
    rng = np.random.RandomState(seed)
    labelled = [c for c in all_cluster_ids if cluster_type_counts.get(c)]
    unlabelled = [c for c in all_cluster_ids if not cluster_type_counts.get(c)]
    rng.shuffle(labelled)
    rng.shuffle(unlabelled)

    types = sorted({t for d in cluster_type_counts.values() for t in d})
    total_per_type = {
        t: sum(d.get(t, 0) for d in cluster_type_counts.values()) for t in types
    }
    target_holdout = {t: total_per_type[t] * holdout_fraction for t in types}
    running_holdout = {t: 0 for t in types}
    assignment: dict[int, str] = {}

    def deficit(running):
        return sum(max(0.0, target_holdout[t] - running.get(t, 0)) ** 2 for t in types)

    n_holdout_target = round(len(all_cluster_ids) * holdout_fraction)
    n_holdout = 0
    for cid in labelled:
        counts = cluster_type_counts[cid]
        trial = dict(running_holdout)
        for t, c in counts.items():
            trial[t] = trial.get(t, 0) + c
        prefer_holdout = deficit(trial) < deficit(running_holdout)
        if n_holdout >= n_holdout_target * 1.2:
            prefer_holdout = False
        assignment[cid] = "holdout" if prefer_holdout else "discovery"
        if prefer_holdout:
            n_holdout += 1
            running_holdout = trial

    for cid in unlabelled:
        prefer_holdout = n_holdout < n_holdout_target
        assignment[cid] = "holdout" if prefer_holdout else "discovery"
        if prefer_holdout:
            n_holdout += 1

    return assignment


def build_corpus(
    swissprot_tsv: Path,
    m1_raw_paths: dict[str, Path],
    parsers: dict[str, Callable],
    cfg,
    tmp_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    corpus = filter_swissprot(swissprot_tsv, cfg.max_protein_length)

    # Raw source parsing runs before CD-HIT (not after), so a parsing bug fails in seconds
    # instead of after paying CD-HIT's full clustering cost.
    corpus_seq = dict(zip(corpus["accession"], corpus["sequence"], strict=False))
    raw_hits = raw_type_hits_by_accession(
        m1_raw_paths, parsers, cfg, corpus_seq=corpus_seq
    )

    cluster_of = run_cdhit(corpus, identity=cfg.cdhit_identity, tmp_dir=tmp_dir)
    corpus["cluster_id"] = corpus["accession"].map(cluster_of)
    n_unclustered = corpus["cluster_id"].isna().sum()
    if n_unclustered:
        raise RuntimeError(
            f"{n_unclustered} proteins did not receive a CD-HIT cluster id"
        )
    corpus["cluster_id"] = corpus["cluster_id"].astype(int)

    cluster_type_counts: dict[int, dict[str, int]] = {}
    for _, row in corpus.iterrows():
        acc, cid = row["accession"], row["cluster_id"]
        if acc in raw_hits:
            d = cluster_type_counts.setdefault(cid, {})
            for t, n in raw_hits[acc].items():
                d[t] = d.get(t, 0) + n

    all_cluster_ids = sorted(corpus["cluster_id"].unique().tolist())
    split_of_cluster = iterative_stratified_split(
        cluster_type_counts, all_cluster_ids, cfg.holdout_fraction, cfg.holdout_seed
    )
    corpus["split"] = corpus["cluster_id"].map(split_of_cluster)

    n_proteins = len(corpus)
    all_types = sorted({t for d in cluster_type_counts.values() for t in d})
    per_type_discovery_sites = {}
    for t in cfg.primary_types:
        disc = sum(
            d.get(t, 0)
            for cid, d in cluster_type_counts.items()
            if split_of_cluster[cid] == "discovery"
        )
        per_type_discovery_sites[t] = disc

    gate_report = {
        "n_proteins": n_proteins,
        "gate_protein_count_15k_20k": 15000 <= n_proteins <= 20000,
        "n_types_stratified_on": len(all_types),
        "per_type_discovery_raw_sites": per_type_discovery_sites,
        "target_holdout_fraction": cfg.holdout_fraction,
        "note": "raw/unfiltered counts for gating the SPLIT only; real counts come from M3.",
    }
    return corpus, gate_report


def save_corpus_artefacts(
    corpus: pd.DataFrame,
    stratum_counts: dict,
    gate_report: dict,
    paths: CorpusPaths,
    cfg,
    fp: str | None = None,
):
    corpus.to_parquet(paths.corpus_parquet, index=False)
    if fp is not None:
        paths.corpus_parquet.with_suffix(
            paths.corpus_parquet.suffix + ".fp"
        ).write_text(fp)
    paths.stratum_counts.write_text(json.dumps(stratum_counts, indent=2))

    split_manifest = {
        "seed": cfg.holdout_seed,
        "holdout_fraction": cfg.holdout_fraction,
        "stratify_by": cfg.holdout_stratify_by,
        "cluster_to_split": corpus.drop_duplicates("cluster_id")
        .set_index("cluster_id")["split"]
        .to_dict(),
        "gate_report": gate_report,
    }
    paths.split_manifest.write_text(json.dumps(split_manifest, indent=2, default=str))
    return paths.corpus_parquet, paths.stratum_counts, paths.split_manifest


def run_m2(
    swissprot_tsv: Path,
    m1_raw_paths: dict[str, Path],
    parsers: dict[str, Callable],
    cfg,
    paths: CorpusPaths,
    tmp_dir: Path | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """corpus.parquet + stratum_counts.json + split_manifest.json are cached and invalidated
    together as one unit, since they must stay mutually consistent."""
    raw_hashes = {
        label: sha256_file(p) for label, p in m1_raw_paths.items() if p.exists()
    }
    expected_fp = config_fingerprint(
        {
            "swissprot_sha256": sha256_file(swissprot_tsv)
            if swissprot_tsv.exists()
            else None,
            "raw_source_hashes": raw_hashes,
        },
        cfg=cfg,
    )

    if not force:
        hit = find_cached("corpus.parquet", search_dirs=[paths.processed_dir])
        if hit is not None:
            fp_sidecar = hit.with_suffix(hit.suffix + ".fp")
            if fp_sidecar.exists() and fp_sidecar.read_text().strip() == expected_fp:
                corpus = pd.read_parquet(paths.corpus_parquet)
                gate_report = json.loads(paths.split_manifest.read_text())[
                    "gate_report"
                ]
                print(
                    f"[M2] CACHE HIT: reused corpus.parquet + stratum_counts.json + split_manifest.json, skipped CD-HIT on {len(corpus)} proteins."
                )
                return corpus, gate_report, {"cached": True, "fingerprint": expected_fp}
            print(
                "[M2] found a cached corpus.parquet but its fingerprint is stale -- rebuilding, including CD-HIT."
            )

    print("[M2] no valid cache -- building corpus from scratch (this runs CD-HIT).")
    corpus, gate_report = build_corpus(
        swissprot_tsv, m1_raw_paths, parsers, cfg, tmp_dir=tmp_dir
    )
    stratum_counts = compute_stratum_counts(corpus, cfg.stratum_residues)
    save_corpus_artefacts(
        corpus, stratum_counts, gate_report, paths, cfg, fp=expected_fp
    )
    return corpus, gate_report, {"cached": False, "fingerprint": expected_fp}
