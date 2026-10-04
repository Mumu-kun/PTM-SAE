"""Runtime environment awareness: which platform we run on, where the repo lives, where data goes.

The same code and YAML configs run on a local PC, a persistent remote GPU box, Kaggle and Colab.
This module is the single place that knows how those differ.
"""

import importlib.metadata
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

Platform = Literal["kaggle", "colab", "local"]

PROJECT_NAME = "ptm-sae-engine"
DATA_ROOT_ENV = "PTM_SAE_DATA_ROOT"
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Cloud runtimes ship a CUDA-matched build of these; letting pip re-resolve them from
# pyproject.toml would silently swap in a generic (non-GPU) wheel.
CLOUD_PREINSTALLED = frozenset({"torch", "transformers"})


def detect_platform() -> Platform:
    """Kaggle and Colab are recognised by their runtime markers; everything else (a Windows
    PC, a persistent Linux remote) is "local"."""
    if Path("/kaggle").exists() or "kaggle_secrets" in sys.modules:
        return "kaggle"
    if Path("/content").exists() or "google.colab" in sys.modules:
        return "colab"
    return "local"


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Walks up from `start` (default: cwd) to the directory whose pyproject.toml declares this
    project, so a notebook kernel started in `notebooks/` still finds the checkout root."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            with open(pyproject, "rb") as f:
                name = tomllib.load(f).get("project", {}).get("name")
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if name == PROJECT_NAME:
            return candidate
    return None


def resolve_data_root() -> Path:
    """Directory that relative data paths (activation cache, corpus, checkpoints) hang off.

    Precedence: PTM_SAE_DATA_ROOT env var, then the platform's writable area (Kaggle's
    /kaggle/working), then the repo root (or cwd if run outside a checkout).
    """
    if env_root := os.environ.get(DATA_ROOT_ENV):
        return Path(env_root).expanduser().resolve()
    if detect_platform() == "kaggle":
        return KAGGLE_WORKING_DIR
    return find_repo_root() or Path.cwd().resolve()


def project_requirements(
    root: str | Path | None = None, extras: tuple[str, ...] = ()
) -> list[str]:
    """Requirement strings from pyproject.toml's runtime dependencies plus the named optional
    extras, so notebooks install exactly what the project declares instead of a hand-kept list."""
    root = Path(root) if root else find_repo_root()
    if root is None:
        raise FileNotFoundError("Not inside a ptm-sae-engine checkout.")
    with open(root / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]
    optional = project.get("optional-dependencies", {})
    return [*project["dependencies"], *(r for e in extras for r in optional[e])]


def requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]


def cloud_requirements(
    root: str | Path | None = None, extras: tuple[str, ...] = ()
) -> list[str]:
    """`project_requirements` minus the packages cloud runtimes pre-install."""
    return [
        r
        for r in project_requirements(root, extras)
        if requirement_name(r).lower() not in CLOUD_PREINSTALLED
    ]


def missing_requirements(
    root: str | Path | None = None, extras: tuple[str, ...] = ()
) -> list[str]:
    """Names of declared requirements not installed in this interpreter's environment."""
    missing = []
    for requirement in project_requirements(root, extras):
        name = requirement_name(requirement)
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return missing


def anchor_path(path: str | Path, root: str | Path) -> str:
    """Resolves a config path against `root` unless it is already absolute."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else Path(root) / p)


def resolve_secret(name: str, explicit: str | None = None) -> str | None:
    """Discovers a secret across a zero-knowledge cascade, so one notebook/CLI runs anywhere:

    1. Explicit argument.
    2. Environment variable `name`.
    3. Google Colab Secrets.
    4. Kaggle Secrets.

    Returns None if no tier has it; callers add their own further fallbacks.
    """
    if explicit:
        return explicit.strip()

    if env_val := os.environ.get(name):
        return env_val.strip()

    if "google.colab" in sys.modules or Path("/content").exists():
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            if colab_val := userdata.get(name):
                return colab_val.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    if "kaggle_secrets" in sys.modules or Path("/kaggle").exists():
        try:
            from kaggle_secrets import (
                UserSecretsClient,  # type: ignore[import-not-found]
            )

            if kaggle_val := UserSecretsClient().get_secret(name):
                return kaggle_val.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    return None
