"""End-to-end smoke test for the baseline SAE training loop, fully offline against synthetic
SafeTensors fixtures (no network, no real ESM-2 activations)."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
    safetensors.torch.save_file(
        {"activations": shard}, cache_dir / "shard_0000.safetensors"
    )

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
        {
            "uniprot_id": "proteinA",
            "partition": "discovery_train",
            "sequence": "A" * 20,
        },
        {
            "uniprot_id": "proteinB",
            "partition": "discovery_train",
            "sequence": "A" * 15,
        },
        # Sequence deliberately matches the positions used by `_write_rich_ptm_sites` below:
        # 1:K 2:K 3:S 4:S 5:S 6:T 7-10:A.
        {
            "uniprot_id": "proteinC",
            "partition": "discovery_val",
            "sequence": "KKSSSTAAAA",
        },
        # held_out: never read by anything the training-time canaries touch.
        {"uniprot_id": "proteinD", "partition": "held_out", "sequence": "K"},
    ]
    pq.write_table(pa.Table.from_pylist(partitions), corpus_dir / "corpus.parquet")

    return cache_dir, corpus_dir


def _write_rich_ptm_sites(corpus_dir: Path) -> None:
    """A discovery_val positive-site set covering both collapse-check strata (`K`, `ST`) plus a
    multi-label residue (position 5 carries two PTM types) — one row per (protein, position,
    ptm_type). `negative_tier`s are no longer stored here: `load_discovery_val_labels` derives
    them at query time by scanning `corpus.parquet`'s sequence for each stratum's unannotated
    residues (positions 2, 4, 6 below all become automatic `hard` negatives, since proteinC
    carries a positive of the same stratum elsewhere). Also includes one held_out site that must
    never be touched by the training-time canary."""
    rows = [
        {
            "uniprot_id": "proteinC",
            "position": 1,
            "stratum": "K",
            "ptm_type": "ubiquitination",
            "partition": "discovery_val",
        },
        {
            "uniprot_id": "proteinC",
            "position": 3,
            "stratum": "ST",
            "ptm_type": "phosphorylation",
            "partition": "discovery_val",
        },
        {
            "uniprot_id": "proteinC",
            "position": 5,
            "stratum": "ST",
            "ptm_type": "phosphorylation",
            "partition": "discovery_val",
        },
        {
            "uniprot_id": "proteinC",
            "position": 5,
            "stratum": "ST",
            "ptm_type": "glycosylation",
            "partition": "discovery_val",
        },
        # A held_out site must never be touched by the training-time canary.
        {
            "uniprot_id": "proteinD",
            "position": 1,
            "stratum": "K",
            "ptm_type": "ubiquitination",
            "partition": "held_out",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), corpus_dir / "labels_stratified.parquet")


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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
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
    """Verify the JumpReLU training loop (LR schedule, tanh+pre-act sparsity ramp) runs end to end."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="jumprelu",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        sparsity_coefficient=20.0,
        total_steps=6,
        warmup_steps=2,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert (checkpoint_dir / "latest" / "config.json").exists()


def test_jumprelu_requires_sparsity_coefficient():
    """Verify the config refuses to silently default sparsity_coefficient for JumpReLU runs."""
    try:
        SAETrainingConfig(sae_type="jumprelu", total_steps=10)
        raise AssertionError("expected ValueError for missing sparsity_coefficient")
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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
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
    assert "tok/s" in out
    assert "ETA" in out


def test_collapse_check_runs_end_to_end(tmp_path, capsys):
    """Verify the opt-in Tier 3 Residue Collapse canary joins labels_stratified.parquet to
    discovery_val activations and logs all three breakdowns (by_stratum, by_ptm_type,
    by_negative_tier), without touching held_out."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    _write_rich_ptm_sites(corpus_dir)
    checkpoint_dir = tmp_path / "checkpoints"

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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)
    out = capsys.readouterr().out

    assert result["final_step"] == 4
    assert "by_stratum/K" in out
    assert "by_ptm_type/ST__phosphorylation" in out
    assert "by_negative_tier/ST" in out


def test_collapse_check_wandb_logging_runs_offline(tmp_path, monkeypatch):
    """Verify the wandb.Table summary + per-section wandb.plot.line_series trend charts (built
    from the collapse-check breakdowns) actually construct and log without error — run in
    WANDB_MODE=offline so no network/credentials are needed, but every wandb API call for real."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    _write_rich_ptm_sites(corpus_dir)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=8,
        batch_size=8,
        eval_interval_steps=4,
        enable_collapse_check=True,
        collapse_check_interval_steps=4,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
        wandb={"enabled": True, "project": "ptm-sae-test"},
    )

    # Runs the collapse-check block twice (step 4 and step 8), exercising the trend-history
    # accumulation across more than one call, not just a single-point chart.
    result = run_sae_training(config)

    assert result["final_step"] == 8


def test_ptm_concentration_check_reports_three_sections(tmp_path):
    """Unit-level check of run_ptm_concentration_check's structure against a label set with a
    multi-label residue (contributes to two ptm_type pairs at once) and a hard negative."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    _write_rich_ptm_sites(corpus_dir)

    from ptm_sae.models import TopKSAEConfig, TopKSAEModel
    from ptm_sae.training.collapse_check import (
        load_discovery_val_labels,
        run_ptm_concentration_check,
    )

    model = TopKSAEModel(TopKSAEConfig(d_in=HIDDEN_DIM, d_hidden=16, k=4))
    labels = load_discovery_val_labels(corpus_dir=str(corpus_dir))
    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=1,
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
    )

    result = run_ptm_concentration_check(model, labels, config, torch.device("cpu"))

    assert set(result.keys()) == {"by_stratum", "by_ptm_type", "by_negative_tier"}
    assert set(result["by_stratum"].keys()) == {"K", "ST"}
    assert set(result["by_ptm_type"].keys()) == {
        "K__ubiquitination",
        "ST__phosphorylation",
        "ST__glycosylation",
    }
    # Both strata carry a positive elsewhere on proteinC, so their unannotated same-stratum
    # residues (positions 2, 4, 6) are automatically derived as `hard` negatives at query time;
    # only `hard` negatives produce a shortcut-suspect flag.
    assert set(result["by_negative_tier"].keys()) == {"K", "ST"}
    assert result["by_negative_tier"]["ST"]["n_latents_shortcut_suspect"] >= 0


def test_residue_dominance_check_runs_end_to_end(tmp_path, capsys):
    """Verify the opt-in, PTM-label-free Residue-Dominance Gate canary runs against
    corpus.parquet sequences alone and logs a collapse_rate."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=4,
        batch_size=8,
        eval_interval_steps=4,
        enable_residue_dominance_check=True,
        collapse_check_interval_steps=4,
        residue_dominance_top_k=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)
    out = capsys.readouterr().out

    assert result["final_step"] == 4
    assert "residue_dominance: collapse_rate=" in out


def test_residue_dominance_accumulator_reports_valid_collapse_rate(tmp_path):
    """Unit-level check that run_residue_dominance_check reports a well-formed collapse_rate
    over the fixture's discovery_val sequence."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)

    from ptm_sae.models import TopKSAEConfig, TopKSAEModel
    from ptm_sae.training.residue_dominance import (
        load_discovery_val_sequences,
        run_residue_dominance_check,
    )

    model = TopKSAEModel(TopKSAEConfig(d_in=HIDDEN_DIM, d_hidden=16, k=4))
    sequences = load_discovery_val_sequences(corpus_dir=str(corpus_dir))
    assert sequences == {"proteinC": "KKSSSTAAAA"}

    config = SAETrainingConfig(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        total_steps=1,
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
    )

    stats = run_residue_dominance_check(
        model, sequences, config, torch.device("cpu"), top_k=3, threshold=0.7
    )

    assert stats["n_latents"] == 16
    assert 0.0 <= stats["collapse_rate"] <= 1.0


def test_gated_training_loop_logs_free_architecture_metrics(tmp_path, capsys):
    """Verify Gated's already-computed l1_loss/aux_loss reach the log without any new
    computation being added."""
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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    run_sae_training(config)
    out = capsys.readouterr().out

    assert "l1_loss=" in out
    assert "aux_loss=" in out


def test_batchtopk_training_loop_logs_running_threshold(tmp_path, capsys):
    """Verify BatchTopK's running_threshold buffer (used for eval-time inference) reaches
    the log."""
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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    run_sae_training(config)
    out = capsys.readouterr().out

    assert "running_threshold=" in out
    assert "aux_loss=" in out


def test_jumprelu_training_loop_logs_threshold_stats(tmp_path, capsys):
    """Verify JumpReLU's per-latent threshold mean/std reach the log."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    config = SAETrainingConfig(
        sae_type="jumprelu",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        sparsity_coefficient=20.0,
        total_steps=6,
        warmup_steps=2,
        batch_size=8,
        eval_interval_steps=3,
        checkpoint_dir=str(checkpoint_dir),
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    run_sae_training(config)
    out = capsys.readouterr().out

    assert "threshold_mean=" in out
    assert "threshold_std=" in out
    assert "decoder_norm_mean=" in out


def test_resume_from_continues_step_count_and_optimizer_state(tmp_path):
    """Verify resume_from picks up step/epoch/dead-latent-census from a prior run's
    checkpoint_dir/latest/ instead of restarting at 0, and that the optimizer isn't cold
    (its state_dict carries over Adam's per-parameter step counts). wandb stays disabled here
    since this test only covers the local resume mechanics, not W&B credentials."""
    cache_dir, corpus_dir = _write_fixture(tmp_path)
    first_checkpoint_dir = tmp_path / "checkpoints_run1"

    base_kwargs = dict(
        sae_type="topk",
        d_in=HIDDEN_DIM,
        d_hidden=16,
        k=4,
        batch_size=8,
        eval_interval_steps=3,
        cache_dir=str(cache_dir),
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    first_config = SAETrainingConfig(
        **base_kwargs, total_steps=3, checkpoint_dir=str(first_checkpoint_dir)
    )
    first_result = run_sae_training(first_config)
    assert first_result["final_step"] == 3
    assert (first_checkpoint_dir / "latest" / "resume_state.pt").exists()

    resume_state = torch.load(
        first_checkpoint_dir / "latest" / "resume_state.pt", weights_only=False
    )
    assert resume_state["step"] == 3
    # Adam tracks a per-parameter step count in its state_dict; after 3 real optimizer.step()
    # calls it should not be the cold "state" dict an unstepped optimizer would have.
    assert len(resume_state["optimizer_state"]["state"]) > 0

    second_checkpoint_dir = tmp_path / "checkpoints_run2"
    resumed_config = SAETrainingConfig(
        **base_kwargs,
        total_steps=6,
        checkpoint_dir=str(second_checkpoint_dir),
        resume_from=str(first_checkpoint_dir),
    )
    resumed_result = run_sae_training(resumed_config)

    assert resumed_result["final_step"] == 6
    assert (second_checkpoint_dir / "latest" / "resume_state.pt").exists()
    final_state = torch.load(
        second_checkpoint_dir / "latest" / "resume_state.pt", weights_only=False
    )
    assert final_state["step"] == 6


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
        partition_folders=False,  # the fixture writes one mixed-layout shard set
        remote_repo_id=None,
        remote_corpus_repo_id=None,
        remote_subpath=None,
        corpus_dir=str(corpus_dir),
        dead_latent_window_tokens=1000,
    )

    result = run_sae_training(config)

    assert result["final_step"] == 6
    assert (checkpoint_dir / "latest" / "config.json").exists()
