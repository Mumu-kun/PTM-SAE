"""scripts/kaggle_run.py: the parts that need no network (metadata and mode injection)."""

import importlib.util
import json
import re
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


def test_default_output_pattern_takes_result_files_and_skips_checkpoints_and_the_data_cache(kaggle_run):
    pattern = re.compile(kaggle_run.RESULT_FILES)
    wanted = ["sweeps/topk_lr_probe/run_state.json", "sweeps/w4096_lr4e-4/metrics.jsonl", "logs/sweep.log", "benchmark.json", "sweeps/bench_c1/state.json"]
    skipped = ["sweeps/w4096_lr4e-4/latest/resume_state.pt", "sweeps/w4096_lr4e-4/best/model.safetensors", "cache/shards/shard_000.safetensors", "sweeps/w4096_lr4e-4/best/config.json"]

    assert all(pattern.search(path) for path in wanted)
    assert not any(pattern.search(path) for path in skipped)


def test_metadata_attaches_the_secrets_dataset_only_when_asked(kaggle_run):
    assert kaggle_run.build_metadata("someone", "x-y", "n.ipynb")["dataset_sources"] == []
    attached = kaggle_run.build_metadata("someone", "x-y", "n.ipynb", secrets_dataset="ptm-sae-secrets")
    assert attached["dataset_sources"] == ["someone/ptm-sae-secrets"]


def test_read_secrets_prefers_the_environment_over_the_env_file_and_skips_unknown_names(kaggle_run, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nGH_TOKEN="from_file"\nHF_TOKEN=hf_file\nUNRELATED=x\n', encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "from_env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert kaggle_run.read_secrets(env_file) == {"GH_TOKEN": "from_env", "HF_TOKEN": "hf_file"}
    assert kaggle_run.read_secrets(tmp_path / "missing.env") == {"GH_TOKEN": "from_env"}
