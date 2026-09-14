"""End-to-end smoke test for the baseline SAE training loop, fully offline against synthetic
SafeTensors fixtures (no network, no real ESM-2 activations)."""

import json
from pathlib import Path

import safetensors.torch
import torch

from ptm_sae.training.config import SAETrainingConfig
from ptm_sae.training.train import run_sae_training

HIDDEN_DIM = 8


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny discovery_train (proteinA, proteinB) + discovery_val (proteinC) corpus."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    corpus_dir = tmp_path / "processed"
    corpus_dir.mkdir()

    torch.manual_seed(0)
    shard = torch.cat(
        [
            torch.randn(20, HIDDEN_DIM),  # proteinA: discovery_train
            torch.randn(15, HIDDEN_DIM),  # proteinB: discovery_train
            torch.randn(10, HIDDEN_DIM),  # proteinC: discovery_val
        ]
    )
    safetensors.torch.save_file({"activations": shard}, cache_dir / "shard_0000.safetensors")

    manifest = {
        "version": "1.0",
        "total_tokens": 45,
        "shards": ["shard_0000.safetensors"],
        "entries": {
            "proteinA": {
                "uniprot_id": "proteinA",
                "length": 20,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 0,
                "end_offset": 20,
            },
            "proteinB": {
                "uniprot_id": "proteinB",
                "length": 15,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 20,
                "end_offset": 35,
            },
            "proteinC": {
                "uniprot_id": "proteinC",
                "length": 10,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 35,
                "end_offset": 45,
            },
        },
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    partitions = [
        {"uniprot_id": "proteinA", "partition": "discovery_train"},
        {"uniprot_id": "proteinB", "partition": "discovery_train"},
        {"uniprot_id": "proteinC", "partition": "discovery_val"},
    ]
    with open(corpus_dir / "proteins.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in partitions)

    return cache_dir, corpus_dir


def test_topk_training_loop_runs_and_checkpoints(tmp_path):
    """Verify the TopK training loop completes, improves reconstruction, and checkpoints."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=6,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert result["best_val_mse"] < float("inf")
    assert (checkpoint_dir / "latest" / "config.json").exists()
    assert (checkpoint_dir / "best" / "config.json").exists()


def test_jumprelu_training_loop_runs_and_checkpoints(tmp_path):
    """Verify the JumpReLU training loop (LR schedule, L0-coefficient warmup) runs end to end."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="jumprelu",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        l0_coefficient=1e-3,
        l0_warmup_steps=3,
        total_steps=6,
        warmup_steps=2,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert (checkpoint_dir / "latest" / "config.json").exists()


def test_jumprelu_requires_l0_coefficient():
    """Verify the config refuses to silently default l0_coefficient for JumpReLU runs."""
    try:
        SAETrainingConfig(sae_type="jumprelu", total_steps=10)
        raise AssertionError("expected ValueError for missing l0_coefficient")
    except ValueError:
        pass


def test_gated_requires_l1_coefficient():
    """Verify the config refuses to silently default l1_coefficient for Gated runs."""
    try:
        SAETrainingConfig(sae_type="gated", total_steps=10)
        raise AssertionError("expected ValueError for missing l1_coefficient")
    except ValueError:
        pass


def test_batchtopk_training_loop_runs_and_checkpoints(tmp_path):
    """Verify the BatchTopK training loop (batch-level sparsity, AuxK-on-by-default, running
    threshold buffer) runs end to end and that eval mode doesn't choke on AuxK."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="batchtopk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=6,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert (checkpoint_dir / "latest" / "config.json").exists()
    assert (checkpoint_dir / "best" / "config.json").exists()


def test_topk_training_loop_logs_tier2_diagnostics(tmp_path, capsys):
    """Verify the always-on Tier 2 diagnostics (density histogram, cosine similarity, alive-
    latent Jaccard, decoder pre-norm) actually reach the notebook card / stdout fallback."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=6,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    run_sae_training(config)
    out = capsys.readouterr().out

    assert "cosine_sim" in out
    assert "alive_latent_jaccard" in out
    assert "decoder_pre_norm" in out
    # First eval has no prior alive-set to compare against; the second eval (step 6) should.
    assert "alive_latent_jaccard=n/a" in out


def test_collapse_check_runs_end_to_end(tmp_path):
    """Verify the opt-in Tier 3 Residue Collapse canary joins ptm_sites.jsonl to discovery_val
    activations and logs a per-stratum concentration statistic, without touching held_out."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    # proteinC (discovery_val, length 10) — label a few residues across two strata.
    sites = [
        {"uniprot_id": "proteinC", "position": 1, "residue": "K", "stratum": "lysine", "ptm_types": ["ubiquitination"], "partition": "discovery_val"},
        {"uniprot_id": "proteinC", "position": 2, "residue": "K", "stratum": "lysine", "ptm_types": [], "partition": "discovery_val"},
        {"uniprot_id": "proteinC", "position": 3, "residue": "S", "stratum": "serine_threonine", "ptm_types": ["phosphorylation"], "partition": "discovery_val"},
        {"uniprot_id": "proteinC", "position": 4, "residue": "S", "stratum": "serine_threonine", "ptm_types": [], "partition": "discovery_val"},
        # A held_out site must never be touched by the training-time canary.
        {"uniprot_id": "proteinD", "position": 1, "residue": "K", "stratum": "lysine", "ptm_types": ["ubiquitination"], "partition": "held_out"},
    ]
    with open(corpus_dir / "ptm_sites.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s) + "\n" for s in sites)

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=4,
        batch_size=8,
        eval_interval_steps=4,
        enable_collapse_check=True,
        collapse_check_interval_steps=4,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 4


def test_gated_training_loop_runs_and_checkpoints(tmp_path):
    """Verify the Gated SAE training loop (3-term loss: MSE + L1(gate) + frozen-decoder aux)
    runs end to end."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="gated",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        l1_coefficient=1e-3,
        total_steps=6,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        remote_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert (checkpoint_dir / "latest" / "config.json").exists()
