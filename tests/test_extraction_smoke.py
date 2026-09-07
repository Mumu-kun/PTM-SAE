"""Permanent Invariant Contract Test: ESM-2 Hook -> SafeTensors Shard -> Zero-Copy Read."""

from pathlib import Path

import pytest
import torch

from ptm_sae.config import PipelineConfig
from ptm_sae.data import parse_uniprot_fasta
from ptm_sae.extraction.extractor import EsmExtractor
from ptm_sae.extraction.pipeline import run_extraction_pipeline
from ptm_sae.extraction.reader import SafeTensorsReader
from ptm_sae.extraction.sharder import SafeTensorsSharder


def test_esm_extraction_and_sharded_read_smoke(tmp_path):
    # 1. Parse sample FASTA
    fasta_path = Path("data/sample.fasta")
    assert fasta_path.exists(), "Sample FASTA file missing."
    records, skipped = parse_uniprot_fasta(fasta_path, max_sequence_length=1022)
    assert len(records) == 3, f"Expected 3 records, got {len(records)}"
    assert len(skipped) == 0, f"Expected 0 skipped, got {len(skipped)}"

    expected_lengths = {
        "P04637": 393,
        "P68431": 136,
        "P62979": 156,
    }
    total_expected_tokens = sum(expected_lengths.values())
    for r in records:
        assert r.uniprot_id in expected_lengths
        assert r.length == expected_lengths[r.uniprot_id]

    # 2. Load dev config and override output_dir to tmp_path
    cfg = PipelineConfig.from_yaml("configs/dev_8m.yaml")
    cfg.sharding.output_dir = str(tmp_path / "cache")
    cfg.sharding.max_shard_bytes = 200_000  # Forces rotation into multiple shards

    # 3. Initialize Extractor and Sharder
    extractor = EsmExtractor(cfg.model, cfg.extraction)
    sharder = SafeTensorsSharder(cfg.sharding)

    assert extractor.hidden_dim == 320, (
        f"ESM2-8M hidden dim should be 320, got {extractor.hidden_dim}"
    )

    # 4. Extract in batches of 2
    batch_size = cfg.extraction.batch_size
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        for out in extractor.extract_batch(batch):
            sharder.add_protein(
                out.uniprot_id,
                out.residue_tensor,
                out.mean_pooled_vector,
            )

    sharder.close()

    # 5. Verify Manifest
    manifest_path = Path(cfg.sharding.output_dir) / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not created"
    assert sharder.manifest["total_tokens"] == total_expected_tokens
    assert len(sharder.manifest["entries"]) == 3
    # Shard rotation should have triggered with 200k bytes threshold
    assert len(sharder.manifest["shards"]) >= 2

    # 6. Zero-copy Readback & Biological Coordinate Invariants
    reader = SafeTensorsReader(cfg.sharding.output_dir)
    assert reader.total_tokens == total_expected_tokens

    # Verify P53 (393 residues)
    p53_acts = reader.get_protein_activations("P04637")
    assert p53_acts.shape == (393, 320)

    # Biological residue coordinate 1 (first residue)
    res_1 = reader.get_residue_activation("P04637", 1)
    assert res_1.shape == (320,)
    assert torch.equal(res_1, p53_acts[0])

    # Biological residue coordinate 393 (last residue)
    res_last = reader.get_residue_activation("P04637", 393)
    assert res_last.shape == (320,)
    assert torch.equal(res_last, p53_acts[392])

    # Boundary out of bounds check
    with pytest.raises(IndexError):
        reader.get_residue_activation("P04637", 0)
    with pytest.raises(IndexError):
        reader.get_residue_activation("P04637", 394)

    # Mean-pooled embedding check
    mean_emb = reader.get_mean_pooled_embedding("P04637")
    assert mean_emb is not None
    assert mean_emb.shape == (320,)
    assert torch.allclose(mean_emb, p53_acts.mean(dim=0), atol=1e-5)

    # 7. Resumption Test: Re-creating sharder recognizes committed proteins
    resuming_sharder = SafeTensorsSharder(cfg.sharding)
    for u_id in expected_lengths:
        assert resuming_sharder.is_committed(u_id)


def test_run_extraction_pipeline_end_to_end(tmp_path):
    cfg = PipelineConfig.from_yaml("configs/dev_8m.yaml")
    cfg.sharding.output_dir = str(tmp_path / "pipeline_cache")
    cfg.sharding.max_shard_bytes = 300_000

    # 1. First execution
    manifest = run_extraction_pipeline(config=cfg, fasta_path="data/sample.fasta")
    assert manifest["total_tokens"] == 685
    assert len(manifest["entries"]) == 3

    # 2. Re-execution verifies automatic resumption
    second_manifest = run_extraction_pipeline(
        config=cfg, fasta_path="data/sample.fasta"
    )
    assert second_manifest["total_tokens"] == 685
    assert len(second_manifest["entries"]) == 3
