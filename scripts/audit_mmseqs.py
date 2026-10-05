"""Independent homology check of a finished split: MMseqs2 across the same partition boundaries the
CD-HIT audit uses (`ptm_sae.corpus.clustering.AUDIT_BOUNDARIES`). MMseqs2 is alignment-based and more
sensitive than CD-HIT near 40% identity, so a clean result here is evidence the split does not leak.

    uv run python scripts/audit_mmseqs.py --corpus runs/corpus_v6/processed/corpus.parquet \
        --wsl-distro ubuntu --mmseqs "~/tools/mmseqs/bin/mmseqs"

Reports, per boundary, how many query proteins have a hit at >= `--identity` identity under two
conventions: any coverage (what cd-hit-2d effectively counts) and >= 50% of the query covered.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ptm_sae.corpus.clustering import AUDIT_BOUNDARIES

HIT_COLUMNS = "query,target,pident,qcov,tcov"


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    path.write_text("".join(f">{row.uniprot_id}\n{row.sequence}\n" for row in frame.itertuples()))


def count_hit_queries(hits_path: Path, min_identity: float, min_query_coverage: float = 0.0) -> int:
    """Distinct query ids with a hit at >= `min_identity` covering >= `min_query_coverage` of the query."""
    queries = set()
    for line in hits_path.read_text().splitlines():
        query, _target, identity, query_coverage, _target_coverage = line.split("\t")
        if float(identity) / 100 >= min_identity and float(query_coverage) >= min_query_coverage:  # pident is a percentage
            queries.add(query)
    return len(queries)


def to_wsl_path(path: Path) -> str:
    drive, rest = path.resolve().as_posix().split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="corpus.parquet with uniprot_id, sequence, partition")
    parser.add_argument("--mmseqs", default="mmseqs", help="mmseqs executable (inside WSL when --wsl-distro is set)")
    parser.add_argument("--wsl-distro", help="run mmseqs inside this WSL distribution (paths are translated)")
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", type=Path, help="summary JSON (default: next to the corpus)")
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus, columns=["uniprot_id", "sequence", "partition"])
    out_path = args.out or args.corpus.parent.parent / "mmseqs_audit.json"
    work = Path(tempfile.mkdtemp(prefix="mmseqs_audit_", dir=args.corpus.parent.parent))
    arg_path = to_wsl_path if args.wsl_distro else str
    prefix = ["wsl", "-d", args.wsl_distro, "--", "bash", "-lc"] if args.wsl_distro else []
    summary = {"identity": args.identity, "sensitivity": args.sensitivity, "boundaries": {}}

    for index, (reference_parts, query_part) in enumerate(AUDIT_BOUNDARIES):
        reference = corpus[corpus["partition"].isin(reference_parts)]
        query = corpus[corpus["partition"] == query_part]
        reference_fasta, query_fasta = work / f"ref{index}.fasta", work / f"query{index}.fasta"
        write_fasta(reference, reference_fasta)
        write_fasta(query, query_fasta)

        hits = work / f"hits{index}.tsv"
        command = (
            f"{args.mmseqs} easy-search {arg_path(query_fasta)} {arg_path(reference_fasta)} {arg_path(hits)} "
            f"{arg_path(work / f'tmp{index}')} --min-seq-id {args.identity} -s {args.sensitivity} -c 0 "
            f"--threads {args.threads} --format-output {HIT_COLUMNS}"
        )
        print(f"[mmseqs] {query_part} vs {'+'.join(reference_parts)}: {len(query)} queries, {len(reference)} targets", flush=True)
        result = subprocess.run([*prefix, command] if prefix else command.split(), text=True)  # noqa: S603 -- local binary, fixed arguments
        if result.returncode != 0:
            raise SystemExit(f"mmseqs failed (rc={result.returncode}) on the {query_part} boundary")

        summary["boundaries"][query_part] = {
            "queries": len(query),
            "references": len(reference),
            "queries_with_hit_any_coverage": count_hit_queries(hits, args.identity),
            "queries_with_hit_query_coverage_50": count_hit_queries(hits, args.identity, 0.5),
        }
        print(summary["boundaries"][query_part], flush=True)

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to {out_path}")


if __name__ == "__main__":
    main()
