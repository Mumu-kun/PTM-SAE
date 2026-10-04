"""Unit tests for SAETrainingConfig data-root anchoring and key=value overrides."""

import pytest

from ptm_sae.training.config import SAETrainingConfig


def _config(**kwargs) -> SAETrainingConfig:
    return SAETrainingConfig(total_steps=100, **kwargs)


def test_with_data_root_anchors_relative_paths_only(tmp_path):
    absolute_cache = tmp_path / "fast_disk" / "cache"
    config = _config(
        cache_dir=str(absolute_cache),
        checkpoint_dir="checkpoints/run",
        corpus_dir="data/processed",
        resume_from="checkpoints/previous",
    ).with_data_root(tmp_path / "root")

    root = tmp_path / "root"
    assert config.cache_dir == str(absolute_cache)
    assert config.checkpoint_dir == str(root / "checkpoints" / "run")
    assert config.corpus_dir == str(root / "data" / "processed")
    assert config.resume_from == str(root / "checkpoints" / "previous")


def test_with_data_root_leaves_missing_resume_from_alone(tmp_path):
    assert _config().with_data_root(tmp_path).resume_from is None


def test_with_overrides_parses_yaml_values_and_nested_keys():
    config = _config().with_overrides(
        ["total_steps=2000", "dtype=bf16", "wandb.enabled=true", "wandb.tags=[a, b]"]
    )

    assert config.total_steps == 2000
    assert config.dtype == "bf16"
    assert config.wandb.enabled is True
    assert config.wandb.tags == ["a", "b"]


def test_with_overrides_rederives_collapse_cadence_from_eval_interval():
    config = _config().with_overrides(["eval_interval_steps=100"])
    assert config.collapse_check_interval_steps == 500

    pinned = _config().with_overrides(
        ["eval_interval_steps=100", "collapse_check_interval_steps=300"]
    )
    assert pinned.collapse_check_interval_steps == 300


@pytest.mark.parametrize(
    "bad_override", ["no_equals_sign", "not_a_field=1", "wandb.not_a_field=1", "dtype=fp64"]
)
def test_with_overrides_rejects_bad_input(bad_override):
    with pytest.raises(ValueError):
        _config().with_overrides([bad_override])
