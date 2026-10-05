"""Helpers of the MMseqs2 homology check (the mmseqs binary itself is not needed)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_mmseqs.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_mmseqs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_fasta_has_one_record_per_protein(audit, tmp_path):
    frame = pd.DataFrame({"uniprot_id": ["A", "B"], "sequence": ["MKV", "GGS"]})

    audit.write_fasta(frame, tmp_path / "x.fasta")

    assert (tmp_path / "x.fasta").read_text() == ">A\nMKV\n>B\nGGS\n"


def test_count_hit_queries_counts_distinct_queries_over_the_thresholds(audit, tmp_path):
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "q1\tt1\t45.0\t0.9\t0.8\n"  # pident is a percentage in MMseqs2 output
        "q1\tt2\t50.0\t0.9\t0.8\n"  # a second hit of the same query counts once
        "q2\tt1\t42.0\t0.3\t0.9\n"  # identity ok, query coverage low
        "q3\tt1\t30.0\t1.0\t1.0\n"  # identity too low
    )

    assert audit.count_hit_queries(hits, 0.4) == 2
    assert audit.count_hit_queries(hits, 0.4, 0.5) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="WSL path translation is for Windows drives")
def test_to_wsl_path_translates_a_windows_drive(audit):
    assert audit.to_wsl_path(Path("E:/a/b")).endswith("/a/b")
    assert audit.to_wsl_path(Path("E:/a/b")).startswith("/mnt/")
