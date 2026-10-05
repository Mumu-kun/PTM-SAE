"""scripts/kaggle_run.py: the parts that need no network (metadata and mode injection)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_run.py"
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pipeline.ipynb"


@pytest.fixture(scope="module")
def kaggle_run():
    spec = importlib.util.spec_from_file_location("kaggle_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inject_mode_sets_the_default_and_leaves_the_rest_alone(kaggle_run):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    smoke = kaggle_run.inject_mode(notebook, "smoke")

    sources = ["".join(c["source"]) for c in smoke["cells"]]
    assert sum('os.environ.get("PTM_SAE_MODE", "smoke")' in s for s in sources) == 1
    assert not any('os.environ.get("PTM_SAE_MODE", "real")' in s for s in sources)
    original = ["".join(c["source"]) for c in notebook["cells"]]
    assert [s for s in original if "PTM_SAE_MODE" not in s] == [s for s in sources if "PTM_SAE_MODE" not in s]
    # the input is not mutated
    assert any('os.environ.get("PTM_SAE_MODE", "real")' in s for s in original)


def test_inject_mode_rejects_bad_input(kaggle_run):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="smoke"):
        kaggle_run.inject_mode(notebook, "fast")
    with pytest.raises(ValueError, match="no PTM_SAE_MODE"):
        kaggle_run.inject_mode({"cells": [{"cell_type": "code", "source": ["print(1)"]}]}, "real")


def test_metadata_is_private_internet_on_cpu_by_default(kaggle_run):
    metadata = kaggle_run.build_metadata("someone", "ptm-sae-corpus-build", "kaggle_pipeline.ipynb")

    assert metadata["id"] == "someone/ptm-sae-corpus-build"
    assert metadata["title"] == "ptm sae corpus build"  # slugifies back to the id's slug
    assert (metadata["is_private"], metadata["enable_internet"], metadata["enable_gpu"]) == ("true", "true", "false")
    assert kaggle_run.build_metadata("someone", "x-y", "n.ipynb", gpu=True)["enable_gpu"] == "true"


TRAIN_NOTEBOOK = NOTEBOOK.parent / "train_sae.ipynb"


def test_inject_mode_sets_other_notebook_parameters(kaggle_run):
    notebook = json.loads(TRAIN_NOTEBOOK.read_text(encoding="utf-8"))

    patched = kaggle_run.inject_mode(
        notebook, "real", params={"PTM_SAE_SWEEP_SPEC": "sweeps/topk_sweep.yaml", "PTM_SAE_QUICK": "1"}
    )

    source = "".join("".join(c["source"]) for c in patched["cells"])
    assert 'os.environ.get("PTM_SAE_SWEEP_SPEC", "sweeps/topk_sweep.yaml")' in source
    assert 'os.environ.get("PTM_SAE_QUICK", "1")' in source
    original = "".join("".join(c["source"]) for c in notebook["cells"])
    assert 'os.environ.get("PTM_SAE_SWEEP_SPEC", "")' in original  # the input is not mutated


def test_inject_mode_rejects_a_parameter_the_notebook_does_not_have(kaggle_run):
    notebook = json.loads(TRAIN_NOTEBOOK.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="PTM_SAE_NOPE"):
        kaggle_run.inject_mode(notebook, "real", params={"PTM_SAE_NOPE": "1"})
