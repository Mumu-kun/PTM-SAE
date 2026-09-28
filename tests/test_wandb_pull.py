"""Unit tests for W&B retrieval helpers, mocking wandb.Api()'s run object."""

from unittest.mock import MagicMock

from ptm_sae.analysis.wandb_pull import pull_histogram, pull_scalars, pull_table


def _mock_run(history_rows):
    run = MagicMock()
    run.scan_history.return_value = iter(history_rows)
    return run


def test_pull_scalars_returns_step_indexed_dataframe():
    run = _mock_run(
        [
            {"_step": 50, "train/l0": 2047.3, "train/threshold_std": 2.8e-9},
            {"_step": 100, "train/l0": 2043.2, "train/threshold_std": 9.9e-9},
        ]
    )
    df = pull_scalars(run, ["train/l0", "train/threshold_std"])
    assert list(df.index) == [50, 100]
    assert df.loc[50, "train/l0"] == 2047.3


def test_pull_histogram_explicit_bins():
    run = _mock_run(
        [
            {
                "_step": 1,
                "val/feature_density_histogram": {
                    "_type": "histogram",
                    "bins": [0.0, 0.5, 1.0],
                    "values": [3, 5],
                },
            }
        ]
    )
    counts, edges = pull_histogram(run, "val/feature_density_histogram")
    assert counts.tolist() == [3, 5]
    assert edges.tolist() == [0.0, 0.5, 1.0]


def test_pull_histogram_packed_bins():
    run = _mock_run(
        [
            {
                "_step": 1,
                "val/feature_density_histogram": {
                    "_type": "histogram",
                    "bins": None,
                    "packedBins": {"count": 2, "min": -12, "size": 6.0},
                    "values": [1053, 3043],
                },
            }
        ]
    )
    counts, edges = pull_histogram(run, "val/feature_density_histogram")
    assert counts.tolist() == [1053, 3043]
    assert edges.tolist() == [-12.0, -6.0, 0.0]


def test_pull_table_finds_matching_artifact():
    artifact = MagicMock()
    artifact.name = "run-abc123-collapse_check/summary:v0"
    table = MagicMock()
    table.get_dataframe.return_value = "a-dataframe"
    artifact.get.return_value = table

    run = MagicMock()
    run.logged_artifacts.return_value = [artifact]

    result = pull_table(
        run, "run-abc123-collapse_check/summary", "collapse_check/summary"
    )
    assert result == "a-dataframe"
