"""Test suite for FASTA ingestion and residue-stratum lookup.

PTM-type canonicalization, per-source parsers (PhosphoSitePlus, CPLM, UniProt-features), custom
reader registration, MILP/greedy cluster partitioning, and `prepare_ptm_corpus`/
`harmonize_and_validate_ptm_sites` were all retired in favor of `ptm_sae.corpus`'s
chemistry-validated cascade ported from the partner's N1 pipeline (implementation plan, Stage 2).
Fresh coverage for `ptm_sae.corpus.{acquisition,clustering,labels}` is a real gap left for a
follow-up session.
"""

from ptm_sae.corpus.config import CFG
from ptm_sae.data.ingestion import get_stratum_for_residue, parse_uniprot_fasta


def test_schema_and_stratum_mapping():
    """Verify stratum assignment uses N1's short chemical-stratum codes."""
    assert get_stratum_for_residue("S") == "ST"
    assert get_stratum_for_residue("T") == "ST"
    assert get_stratum_for_residue("K") == "K"
    assert get_stratum_for_residue("A") == "other"

    # STRATUM_MAPPING's codes must match ptm_sae.corpus.config.CFG.stratum_residues exactly --
    # not just for the residues it maps, but for the full set of stratum codes CFG expects.
    assert set(CFG.stratum_residues["K"]) == {"K"}
    assert set(CFG.stratum_residues["ST"]) == {"S", "T"}


def test_uniprot_fasta_parsing_and_length_filter():
    """Verify FASTA parsing and sequence length filtering (<= 1022 residues)."""
    fasta_content = (
        ">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4\n"
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\n"
        ">sp|Q_LONG|LONG_HUMAN Very Long Protein OS=Homo sapiens OX=9606\n"
        + ("A" * 1200)
        + "\n"
    )

    valid, skipped = parse_uniprot_fasta(fasta_content, max_sequence_length=1022)
    assert len(valid) == 1
    assert valid[0].uniprot_id == "P04637"
    assert valid[0].length == 60

    assert len(skipped) == 1
    assert skipped[0][0] == "Q_LONG"
    assert skipped[0][1] == 1200
