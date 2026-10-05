"""Unit tests for platform detection, repo-root discovery, data-root resolution and secrets."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from ptm_sae import runtime


def _fake_project(root: Path, name: str = runtime.PROJECT_NAME) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")
    return root


def _no_cloud_markers(monkeypatch):
    # Path("/kaggle") / Path("/content") probes must not depend on the machine running tests.
    monkeypatch.setattr(runtime.Path, "exists", lambda self: False)
    monkeypatch.delitem(runtime.sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(runtime.sys.modules, "kaggle_secrets", raising=False)


def test_find_repo_root_walks_up_from_nested_dir(tmp_path):
    root = _fake_project(tmp_path / "repo")
    nested = root / "notebooks" / "deep"
    nested.mkdir(parents=True)

    assert runtime.find_repo_root(nested) == root.resolve()


def test_find_repo_root_ignores_other_projects(tmp_path):
    other = _fake_project(tmp_path / "other", name="something-else")

    assert runtime.find_repo_root(other) is None


def test_detect_platform_table(monkeypatch):
    _no_cloud_markers(monkeypatch)
    assert runtime.detect_platform() == "local"

    for marker, expected in (("/kaggle", "kaggle"), ("/content", "colab")):
        monkeypatch.setattr(
            runtime.Path, "exists", lambda self, m=marker: self.as_posix() == m
        )
        assert runtime.detect_platform() == expected


def test_resolve_data_root_precedence(tmp_path, monkeypatch):
    repo = _fake_project(tmp_path / "repo")
    monkeypatch.chdir(repo)
    _no_cloud_markers(monkeypatch)

    # 1. Falls back to the repo root locally
    monkeypatch.delenv(runtime.DATA_ROOT_ENV, raising=False)
    assert runtime.resolve_data_root() == repo.resolve()

    # 2. Kaggle's writable directory beats the repo root
    monkeypatch.setattr(runtime, "detect_platform", lambda: "kaggle")
    assert runtime.resolve_data_root() == runtime.KAGGLE_WORKING_DIR

    # 3. The env var beats everything
    big_disk = tmp_path / "big_disk"
    monkeypatch.setenv(runtime.DATA_ROOT_ENV, str(big_disk))
    assert runtime.resolve_data_root() == big_disk.resolve()


def test_anchor_path_respects_absolute_paths(tmp_path):
    assert runtime.anchor_path("cache/a", tmp_path) == str(tmp_path / "cache" / "a")

    absolute = tmp_path / "elsewhere"
    assert runtime.anchor_path(absolute, "/ignored") == str(absolute)


def test_resolve_secret_cascade(monkeypatch):
    _no_cloud_markers(monkeypatch)

    # 1. Explicit argument beats the env var
    monkeypatch.setenv("PTM_TEST_SECRET", " from_env ")
    assert runtime.resolve_secret("PTM_TEST_SECRET", "explicit") == "explicit"

    # 2. Env var is stripped and used next
    assert runtime.resolve_secret("PTM_TEST_SECRET") == "from_env"

    # 3. Nothing found anywhere returns None
    monkeypatch.delenv("PTM_TEST_SECRET")
    assert runtime.resolve_secret("PTM_TEST_SECRET") is None


def test_requirement_helpers(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """
[project]
name = "ptm-sae-engine"
dependencies = ["torch>=2.2.0", "pydantic>=2.6.0", "PyYAML>=6", "not-a-real-pkg-xyz>=1"]

[project.optional-dependencies]
kaggle = ["kaggle>=1.6"]
""",
        encoding="utf-8",
    )

    assert runtime.requirement_name("scipy[extra]>=1.0; python_version>'3'") == "scipy"
    assert runtime.project_requirements(root, ("kaggle",))[-1] == "kaggle>=1.6"
    cloud = runtime.cloud_requirements(root)
    assert "torch" not in cloud and "torch>=2.2.0" not in cloud  # preinstalled CUDA build is kept
    assert "pydantic" in cloud and "pydantic>=2.6.0" not in cloud  # bare name: no upgrade of the image's version
    assert runtime.missing_requirements(root) == ["not-a-real-pkg-xyz"]


def test_resolve_secret_reads_only_the_detected_platforms_store(monkeypatch):
    class FakeSecretsClient:
        def get_secret(self, name):
            return " from_kaggle " if name == "PTM_TEST_SECRET" else None

    fake_module = types.ModuleType("kaggle_secrets")
    fake_module.UserSecretsClient = FakeSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)
    monkeypatch.delenv("PTM_TEST_SECRET", raising=False)

    # The Kaggle store is consulted on Kaggle (value stripped) ...
    monkeypatch.setattr(runtime, "detect_platform", lambda: "kaggle")
    assert runtime.resolve_secret("PTM_TEST_SECRET") == "from_kaggle"
    assert runtime.resolve_secret("OTHER_SECRET") is None

    # ... and never elsewhere, even if the module happens to be importable.
    monkeypatch.setattr(runtime, "detect_platform", lambda: "local")
    assert runtime.resolve_secret("PTM_TEST_SECRET") is None


def test_importing_the_runtime_module_loads_no_heavy_package():
    """Notebooks import ptm_sae.runtime before installing numpy/torch; that import must not load them."""
    code = (
        "import sys, ptm_sae.runtime\n"
        "heavy = [m for m in ('numpy', 'pandas', 'torch', 'scipy', 'pyarrow') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603

    assert result.returncode == 0, result.stderr


def test_package_entry_points_still_resolve_lazily():
    import ptm_sae

    assert callable(ptm_sae.run_sae_training) and callable(ptm_sae.run_full_lifecycle)
    with pytest.raises(AttributeError):
        ptm_sae.not_a_real_name  # noqa: B018
