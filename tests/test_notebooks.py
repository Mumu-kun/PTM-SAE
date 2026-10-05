"""Static checks that keep the notebooks salvage-proof: valid code, and every stage guarded."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = ("notebooks/kaggle_pipeline.ipynb", "notebooks/train_sae.ipynb", "notebooks/train_sae_k_probe.ipynb", "notebooks/diagnose_activations.ipynb")
REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_JOB_CALLS = ("run_logged(", "upload_to_hf(", "fetch_run(", "publish_kaggle_dataset(")


def _code_cells(name: str) -> list[str]:
    notebook = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _without_ipython_magics(source: str) -> str:
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%")) else line
        for line in source.split("\n")
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_is_valid_python(name):
    for index, source in enumerate(_code_cells(name)):
        compile(_without_ipython_magics(source), f"{name}:cell{index}", "exec")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_stage_after_the_state_is_created_runs_inside_a_guard(name):
    cells = _code_cells(name)
    first = next(i for i, src in enumerate(cells) if "RunState()" in src)

    for index, source in enumerate(cells[first:], start=first):
        assert "with state.cell(" in source or "state.print_summary(" in source, (
            f"{name} code cell {index} runs outside a guard: one exception there would end the run"
        )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_long_jobs_only_run_inside_guarded_cells(name):
    for index, source in enumerate(_code_cells(name)):
        if any(call in source for call in LONG_JOB_CALLS):
            assert "with state.cell(" in source, f"{name} cell {index} runs a long job unguarded"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebooks_are_platform_agnostic(name):
    """The same file must run on the GPU box and on Kaggle: no hard-coded platform paths."""
    for index, source in enumerate(_code_cells(name)):
        # Platform detection lives in ptm_sae.runtime; the one allowed literal is the Kaggle clone dir
        # in the repo-setup cell, which falls back to the cwd everywhere else.
        if "/kaggle/working" in source:
            assert "Path(\"/kaggle\").exists()" in source, f"{name} cell {index} hard-codes a Kaggle path"
        assert "/content/" not in source, f"{name} cell {index} hard-codes a Colab path"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_heavy_package_is_imported_before_the_install_cell(name):
    """pip may upgrade numpy/torch in the install cell; anything imported earlier would stay
    half-old inside the kernel (observed on Kaggle). Cells before the install cell stay light."""
    cells = _code_cells(name)
    install = next(i for i, src in enumerate(cells) if "pip" in src and "cloud_requirements" in src)
    for index, source in enumerate(cells[:install]):
        for heavy in ("import torch", "import numpy", "import pandas", "import pyarrow", "import scipy", "from ptm_sae.training", "from ptm_sae.corpus", "from ptm_sae.data"):
            assert heavy not in source, f"{name} cell {index} imports {heavy!r} before dependencies are installed"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_failed_clone_cannot_print_the_github_token(name):
    """The clone URL embeds the token; subprocess errors repr the whole command line."""
    setup = next(src for src in _code_cells(name) if "x-access-token" in src)

    assert "capture_output=True" in setup and "from None" in setup
    assert "except subprocess.CalledProcessError" in setup


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_clone_cell_reads_the_secrets_dataset_the_runtime_reads(name):
    """The clone cell runs before ptm_sae exists, so it spells the secrets file name itself: it must match."""
    from ptm_sae.runtime import SECRETS_FILE_NAME

    setup = next(src for src in _code_cells(name) if "x-access-token" in src)

    assert f'"*/{SECRETS_FILE_NAME}"' in setup and f'"*/*/{SECRETS_FILE_NAME}"' in setup


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_final_summary_cell_cannot_be_what_fails_a_run(name):
    """After an install failure numpy/pandas can be broken, so the last cell must import nothing from
    the package (observed: it crashed on `from ptm_sae.corpus.config import ...` and ended the run)."""
    summary = _code_cells(name)[-1]

    assert "state.print_summary(" in summary
    assert "ptm_sae" not in summary and "import" not in summary
    assert "try:" in summary and "except Exception" in summary


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cloud_install_tolerates_a_newer_python_than_pyproject_names(name):
    """Kaggle's image moved to Python 3.13 while requires-python stopped at 3.12; the editable install must not care."""
    install = next(src for src in _code_cells(name) if "cloud_requirements" in src)

    assert "--ignore-requires-python" in install


def test_training_notebook_syncs_data_before_any_training_cell():
    cells = _code_cells("notebooks/train_sae.ipynb")

    def index_of(needle: str) -> int:
        return next(i for i, src in enumerate(cells) if f'state.cell("{needle}")' in src)

    assert index_of("data") < index_of("benchmark") < index_of("sweep")
    assert index_of("data") < index_of("train")
    # a sweep replaces the single run, never both
    assert "if SWEEP_SPEC:" in cells[index_of("train")] and "if not SWEEP_SPEC:" in cells[index_of("sweep")]


def test_k_probe_notebook_differs_from_the_training_notebook_only_in_its_parameter_defaults():
    """The k probe is the training notebook with two different defaults (and its own title). Any other difference is
    drift: a fix made to one copy and forgotten in the other."""
    probe_defaults = {
        'os.environ.get("PTM_SAE_SWEEP_SPEC", "sweeps/topk_k_probe.yaml")': 'os.environ.get("PTM_SAE_SWEEP_SPEC", "")',
        'os.environ.get("PTM_SAE_BENCHMARK", "0")': 'os.environ.get("PTM_SAE_BENCHMARK", "1")',
    }
    training = _code_cells("notebooks/train_sae.ipynb")
    probe = _code_cells("notebooks/train_sae_k_probe.ipynb")

    assert len(probe) == len(training)
    for probe_cell, training_cell in zip(probe, training, strict=True):
        for new, old in probe_defaults.items():
            probe_cell = probe_cell.replace(new, old)
        assert probe_cell == training_cell
