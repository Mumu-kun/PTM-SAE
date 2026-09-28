"""Unit tests for the table-driven check registry and verdict/plotting logic."""

import numpy as np
import pandas as pd
import pytest

from ptm_sae.analysis.diagnostics import plot_checks, run_diagnostics

_HEALTHY_SCALARS = pd.DataFrame(
    {
        "val/explained_variance": [0.90],
        "val/cosine_sim_mean": [0.95],
        "val/cosine_sim_p10": [0.90],
        "val/mse": [0.018],
        "train/decoder_pre_norm_mean": [1.0],
        "val/dead_latent_fraction": [0.02],
        "val/alive_latent_jaccard": [0.97],
        "train/threshold_std": [0.01],
        "train/threshold_mean": [0.05],
        "train/l0": [200],
        "train/decoder_norm_mean": [1.4],
    },
    index=[24000],
)

_JUMPRELU_CONFIG = {"d_hidden": 4096, "sparsity_coefficient": 20.0}


_NO_TARGET_METRICS = {"Threshold Std (log-scale)", "Threshold Mean", "Decoder Norm Mean"}


def test_run_diagnostics_all_pass_for_healthy_run():
    result = run_diagnostics(_HEALTHY_SCALARS, "jumprelu", _JUMPRELU_CONFIG)
    common_rows = result.verdict[~result.verdict["metric"].isin(_NO_TARGET_METRICS)]
    assert (common_rows["status"] == "pass").all()


def test_run_diagnostics_skips_decoder_pre_norm_check_for_jumprelu():
    """train/decoder_pre_norm_mean has no unit-norm target for jumprelu (decoder norm is
    deliberately unconstrained under the tanh+pre-act loss) — it must not appear in the
    verdict at all, since its band would misreport a healthy, drifted decoder norm as failing."""
    result = run_diagnostics(_HEALTHY_SCALARS, "jumprelu", _JUMPRELU_CONFIG)
    assert "Decoder Pre-Norm Mean" not in result.verdict["metric"].values


def test_run_diagnostics_flags_dense_collapse_recommendation():
    dense_scalars = _HEALTHY_SCALARS.copy()
    dense_scalars["train/l0"] = 2919  # 71% of d_hidden=4096, mirrors the real bad run
    result = run_diagnostics(dense_scalars, "jumprelu", _JUMPRELU_CONFIG)
    assert any(
        "dense" in obs.lower() or "sparsity_coefficient" in rec.lower()
        for obs, _, rec in result.recommendations
    )


def test_run_diagnostics_flags_decoder_norm_not_moving():
    stuck_scalars = _HEALTHY_SCALARS.copy()
    stuck_scalars["train/decoder_norm_mean"] = 1.001
    result = run_diagnostics(stuck_scalars, "jumprelu", _JUMPRELU_CONFIG)
    assert any("decoder_norm_mean" in obs.lower() for obs, _, _ in result.recommendations)


def test_run_diagnostics_topk_uses_arch_specific_check():
    topk_scalars = _HEALTHY_SCALARS.drop(
        columns=["train/threshold_std", "train/threshold_mean"]
    )
    result = run_diagnostics(topk_scalars, "topk", {"d_hidden": 4096, "k": 32})
    assert "L0 (pinned near k)" in result.verdict["metric"].values


def test_plot_checks_writes_one_png_per_check(tmp_path):
    counts = np.array([0, 0, 1053, 3043])
    edges = np.array([-12.0, -9.0, -6.0, -3.0, 0.0])
    paths = plot_checks(
        _HEALTHY_SCALARS,
        "jumprelu",
        tmp_path,
        histogram=(counts, edges),
    )
    assert paths
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize("sae_type", ["jumprelu", "topk"])
def test_plot_checks_handles_missing_optional_columns(tmp_path, sae_type):
    minimal = _HEALTHY_SCALARS[["val/explained_variance"]]
    paths = plot_checks(minimal, sae_type, tmp_path)
    assert paths == [tmp_path / "val_explained_variance.png"]
