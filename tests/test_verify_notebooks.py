"""scripts/verify_notebooks.py: change detection and run-state evaluation (no network, no kernel)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_notebooks.py"


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_notebooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    for rel, text in {
        "notebooks/a.ipynb": "{}",
        "src/ptm_sae/mod.py": "x = 1\n",
        "configs/c.yaml": "k: 1\n",
        "pyproject.toml": "[project]\n",
        "docs/readme.md": "not watched\n",
        "tests/test_x.py": "not watched\n",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_hash_changes_only_when_watched_code_changes(verify, repo):
    base = verify.watched_hash(repo)

    (repo / "docs" / "readme.md").write_text("edited docs")
    (repo / "tests" / "test_x.py").write_text("edited tests")
    (repo / "src" / "ptm_sae" / "__pycache__").mkdir()
    (repo / "src" / "ptm_sae" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"junk")
    assert verify.watched_hash(repo) == base  # docs, tests and bytecode do not need re-verification

    (repo / "src" / "ptm_sae" / "mod.py").write_text("x = 2\n")
    changed = verify.watched_hash(repo)
    assert changed != base

    (repo / "configs" / "c.yaml").write_text("k: 2\n")
    assert verify.watched_hash(repo) != changed


def test_a_renamed_or_added_file_changes_the_hash(verify, repo):
    base = verify.watched_hash(repo)

    (repo / "src" / "ptm_sae" / "new.py").write_text("")
    with_new = verify.watched_hash(repo)
    (repo / "src" / "ptm_sae" / "new.py").rename(repo / "src" / "ptm_sae" / "renamed.py")

    assert with_new != base and verify.watched_hash(repo) != with_new


def test_is_verified_needs_a_successful_run_for_the_same_hash(verify):
    state = {"kaggle": {"hash": "abc", "ok": True}, "local": {"hash": "abc", "ok": False}}

    assert verify.is_verified(state, "kaggle", "abc")
    assert not verify.is_verified(state, "kaggle", "def")  # code changed since
    assert not verify.is_verified(state, "local", "abc")  # that run failed
    assert not verify.is_verified(state, "missing", "abc")


def test_evaluate_flags_failed_and_missing_required_cells(verify):
    healthy = {name: {"status": "ok", "detail": ""} for name in verify.REQUIRED_OK["train"]}
    healthy["diagnostics_pull"] = {"status": "skipped", "detail": "wandb off"}
    assert verify.evaluate("train", healthy) == []

    broken = {**healthy, "train": {"status": "failed", "detail": "RuntimeError: boom"}}
    broken.pop("readback")
    problems = verify.evaluate("train", broken)
    assert any("train: failed" in p for p in problems)
    assert any("readback did not complete" in p for p in problems)

    skipped_required = {**healthy, "preflight": {"status": "skipped", "detail": ""}}
    assert any("preflight did not complete" in p for p in verify.evaluate("train", skipped_required))
