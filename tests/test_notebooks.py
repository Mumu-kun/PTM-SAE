"""Static checks that keep the notebooks salvage-proof: valid code, and every stage guarded."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = ("notebooks/kaggle_pipeline.ipynb", "notebooks/train_sae.ipynb")
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
