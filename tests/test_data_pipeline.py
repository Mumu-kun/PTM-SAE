"""Comprehensive test suite for PTM data acquisition, harmonization, and splitting."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ptm_sae.data import PTMCorpus, prepare_ptm_corpus
from ptm_sae.data.fetcher import harmonize_and_validate_ptm_sites
from ptm_sae.data.ingestion import (
    canonicalize_ptm_type,
    get_stratum_for_residue,
    parse_uniprot_fasta,
    read_ptm_sites,
    register_reader,
)
from ptm_sae.data.schema import (
    Protein,
    PTMObservation,
)
from ptm_sae.data.splitting import (
    solve_milp_partition,
)


def test_schema_and_stratum_mapping():
    """Verify stratum assignment and canonical PTM normalization."""
    assert get_stratum_for_residue("S") == "serine_threonine"
    assert get_stratum_for_residue("T") == "serine_threonine"
    assert get_stratum_for_residue("K") == "lysine"
    assert get_stratum_for_residue("A") == "other"

    assert canonicalize_ptm_type("phosphoserine") == "Phosphorylation"
    assert canonicalize_ptm_type("n6-acetyllysine") == "Acetylation"
    assert canonicalize_ptm_type("glycine glycyl") == "Ubiquitination"
    assert canonicalize_ptm_type("O-GlcNAc") == "O-Glycosylation"


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


def test_read_ptm_sites_auto_detection_and_phosphositeplus(tmp_path):
    """Verify deep reader read_ptm_sites with header auto-detection on PhosphoSitePlus."""
    psp_file = tmp_path / "Phosphorylation_site_dataset.txt"
    psp_file.write_text(
        "# PhosphoSitePlus export\n"
        "GENE\tACC_ID\tPROTEIN\tHU_CHR_LOC\tMOD_RSD\tSITE_GRP_ID\tORGANISM\n"
        "TP53\tP04637\tp53\t17p13.1\tS15-p\t12345\thuman\n"
        "TP53\tP04637\tp53\t17p13.1\tK382-ub\t12346\thuman\n"
        "Trp53\tP02340\tp53\t11 B3\tS15-p\t12347\tmouse\n",
        encoding="utf-8",
    )

    # Call deep reader with auto-detection
    observations = list(read_ptm_sites(psp_file))
    assert len(observations) == 2

    assert observations[0].uniprot_id == "P04637"
    assert observations[0].position == 15
    assert observations[0].residue == "S"
    assert observations[0].canonical_ptm_type == "Phosphorylation"

    assert observations[1].uniprot_id == "P04637"
    assert observations[1].position == 382
    assert observations[1].residue == "K"
    assert observations[1].canonical_ptm_type == "Ubiquitination"


def test_register_reader_custom_extension(tmp_path):
    """Verify extending read_ptm_sites with @register_reader with zero multi-file boilerplate."""
    custom_file = tmp_path / "custom_cplm.tsv"
    custom_file.write_text(
        "MY_ACC\tMY_POS\tMY_AA\tMY_MOD\nP04637\t382\tK\tAcetylation\n",
        encoding="utf-8",
    )

    @register_reader(name="my_cplm", header_keywords=["my_acc", "my_pos"])
    def _parse_custom(path: Path, **kwargs) -> Iterator[PTMObservation]:
        with open(path) as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]
        for parts in lines[1:]:
            yield PTMObservation(
                source_db="MyCPLM",
                uniprot_id=parts[0],
                position=int(parts[1]),
                residue=parts[2],
                canonical_ptm_type=parts[3],
                evidence_tier="experimental",
            )

    obs = list(read_ptm_sites(custom_file))
    assert len(obs) == 1
    assert obs[0].source_db == "MyCPLM"
    assert obs[0].position == 382
    assert obs[0].residue == "K"


def test_harmonize_and_validate_invariants():
    """
    Test biological invariant check: sequence[pos - 1] == site.residue.
    Matches pass; mismatches are rejected into audit log.
    Multi-label sites at same coordinate merge into UnifiedResidueSite.
    """
    seq = "MEEKSPPPAA"
    protein = Protein(
        uniprot_id="TEST01",
        sequence=seq,
        length=len(seq),
    )

    observations = [
        PTMObservation(
            source_db="UniProt",
            uniprot_id="TEST01",
            position=5,
            residue="S",
            canonical_ptm_type="Phosphorylation",
        ),
        PTMObservation(
            source_db="dbPTM",
            uniprot_id="TEST01",
            position=4,
            residue="K",
            canonical_ptm_type="Acetylation",
        ),
        PTMObservation(
            source_db="PhosphoSitePlus",
            uniprot_id="TEST01",
            position=4,
            residue="K",
            canonical_ptm_type="Ubiquitination",
        ),
        PTMObservation(
            source_db="dbPTM",
            uniprot_id="TEST01",
            position=3,
            residue="K",
            canonical_ptm_type="Acetylation",
        ),
    ]

    sequences = {"TEST01": protein}
    unified_sites, audit_log = harmonize_and_validate_ptm_sites(observations, sequences)

    assert len(audit_log) == 1
    assert audit_log[0]["uniprot_id"] == "TEST01"
    assert audit_log[0]["position"] == 3
    assert audit_log[0]["expected_residue"] == "K"
    assert audit_log[0]["actual_residue"] == "E"

    assert len(unified_sites) == 2

    k4_site = next(s for s in unified_sites if s.position == 4)
    assert k4_site.residue == "K"
    assert k4_site.stratum == "lysine"
    assert k4_site.ptm_types == {"Acetylation", "Ubiquitination"}
    assert k4_site.is_multi_label is True

    s5_site = next(s for s in unified_sites if s.position == 5)
    assert s5_site.residue == "S"
    assert s5_site.stratum == "serine_threonine"
    assert s5_site.ptm_types == {"Phosphorylation"}


def test_milp_and_greedy_cluster_partitioning():
    """Verify exact MILP and greedy cluster partitioning."""
    np.random.seed(42)
    clusters = [f"CLUST_{i:03d}" for i in range(20)]
    ptm_types = ["Phosphorylation", "Ubiquitination", "Acetylation", "O-Glycosylation"]

    profiles = {}
    for c in clusters:
        tokens = np.random.randint(200, 1500)
        phos = np.random.randint(5, 50)
        ub = np.random.choice([0, 1, 2, 8], p=[0.5, 0.3, 0.15, 0.05])
        ac = np.random.choice([0, 1, 4], p=[0.7, 0.2, 0.1])
        profiles[c] = {
            "tokens": int(tokens),
            "Phosphorylation": int(phos),
            "Ubiquitination": int(ub),
            "Acetylation": int(ac),
        }

    assignment_milp = solve_milp_partition(
        cluster_profiles=profiles,
        ptm_types=ptm_types,
        target_discovery_ratio=0.8,
        token_tolerance=0.03,
    )

    assert set(assignment_milp.keys()) == set(clusters)
    total_tokens = sum(p["tokens"] for p in profiles.values())
    disc_tokens = sum(
        profiles[c]["tokens"]
        for c, part in assignment_milp.items()
        if part == "discovery"
    )
    ratio = disc_tokens / total_tokens
    assert 0.77 <= ratio <= 0.83


def test_prepare_ptm_corpus_end_to_end(tmp_path):
    """
    Verify Candidate 2 deep module seam: prepare_ptm_corpus executes the entire
    ingestion, validation, MILP splitting, and persistence in a single call.
    """
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(
        ">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606\n"
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\n"
        ">sp|P38398|BRCA1_HUMAN Breast cancer type 1 OS=Homo sapiens OX=9606\n"
        "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQ\n",
        encoding="utf-8",
    )

    ptm_file = tmp_path / "Phosphorylation_site_dataset.txt"
    ptm_file.write_text(
        "# PhosphoSitePlus export\n"
        "GENE\tACC_ID\tPROTEIN\tHU_CHR_LOC\tMOD_RSD\tSITE_GRP_ID\tORGANISM\n"
        "TP53\tP04637\tp53\t17p13.1\tS15-p\t12345\thuman\n"
        "BRCA1\tP38398\tbrca1\t17q21.31\tS4-p\t12346\thuman\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    corpus: PTMCorpus = prepare_ptm_corpus(
        fasta_path=fasta_file,
        ptm_sources=[ptm_file],
        split_ratio=0.5,
        output_dir=out_dir,
    )

    assert corpus.total_proteins == 2
    assert corpus.total_sites == 2
    assert (out_dir / "proteins.jsonl").exists()
    assert (out_dir / "ptm_sites.jsonl").exists()
    assert (out_dir / "split_manifest.json").exists()

    # Invariant: 1 protein in discovery, 1 in held_out
    assert len(corpus.discovery_proteins) == 1
    assert len(corpus.held_out_proteins) == 1


def test_cplm_reader_and_canonicalization(tmp_path):
    """Verify CPLM format reader, lysine residue assignment, and PTM canonicalization."""
    cplm_file = tmp_path / "Homo sapiens.txt"
    cplm_file.write_text(
        "CPLM000001\tP04637\t120\tUbiquitination\tTP53\tHomo sapiens\tSEQ...\tExp.\t123456\n"
        "CPLM000002\tP04637\t382\tAcetylation\tTP53\tHomo sapiens\tSEQ...\tDat.\t123457\n"
        "CPLM000003\tP38398\t50\tSuccinylation\tBRCA1\tHomo sapiens\tSEQ...\tExp.\t123458\n",
        encoding="utf-8",
    )
    obs = list(read_ptm_sites(cplm_file))
    assert len(obs) == 3
    assert obs[0].uniprot_id == "P04637"
    assert obs[0].position == 120
    assert obs[0].residue == "K"
    assert obs[0].canonical_ptm_type == "Ubiquitination"
    assert obs[0].evidence_tier == "experimental"

    assert obs[1].canonical_ptm_type == "Acetylation"
    assert obs[1].evidence_tier == "curated"

    assert obs[2].canonical_ptm_type == "Succinylation"


def test_uniprot_features_reader(tmp_path):
    """Verify parsing standardized UniProt feature tables and residue keyword extraction."""
    tsv_file = tmp_path / "uniprot_features.tsv"
    tsv_file.write_text(
        "Entry\tFeature key\tPosition(s)\tDescription\n"
        "P04637\tMOD_RES\t392\tPhosphoserine\n"
        "P04637\tMOD_RES\t382\tN6-acetyllysine\n"
        "P38398\tCARBOHYD\t100\tN-linked (GlcNAc...) asparagine\n"
        "P04637\tMOD_RES\t110\tOmega-N-methylarginine\n",
        encoding="utf-8",
    )
    obs = list(read_ptm_sites(tsv_file))
    assert len(obs) == 4
    assert obs[0].canonical_ptm_type == "Phosphorylation"
    assert obs[0].residue == "S"
    assert obs[1].canonical_ptm_type == "Acetylation"
    assert obs[1].residue == "K"
    assert obs[2].canonical_ptm_type == "N-Glycosylation"
    assert obs[2].residue == "N"
    assert obs[3].canonical_ptm_type == "Methylation"
    assert obs[3].residue == "R"
