"""Runtime environment awareness: which platform we run on, where the repo lives, where data goes.

The same code and YAML configs run on a local PC, a persistent remote GPU box, Kaggle and Colab.
This module is the single place that knows how those differ.
"""

import contextlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import urllib.request
from dataclasses import dataclass, field
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

    platform = detect_platform()

    if platform == "colab":
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            if colab_val := userdata.get(name):
                return colab_val.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    if platform == "kaggle":
        try:
            from kaggle_secrets import (
                UserSecretsClient,  # type: ignore[import-not-found]
            )

            if kaggle_val := UserSecretsClient().get_secret(name):
                return kaggle_val.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    return None


# ---------------------------------------------------------------------------
# Salvage-proof notebook runs: guarded cells, logged subprocesses, preflight.
# One bad cell must never invalidate a long Kaggle run, so failures are recorded and later cells
# that depend on them skip with a reason; every long job runs as a subprocess whose exit code is
# checked, never raised.
# ---------------------------------------------------------------------------
RUN_STATE_FILE = "run_state.json"
KAGGLE_SESSION_LIMIT_S = 12 * 3600
SESSION_SAFETY_S = 30 * 60  # long jobs are stopped this long before the platform's hard limit
TIMEOUT_EXIT_CODE = 124  # same convention as coreutils `timeout`
PROGRESS_LINE = re.compile(r"\d+%\||it/s\]|s/it\]")


class CellSkipped(Exception):
    """Raised by `RunState.require` when a cell's prerequisites did not complete."""


class RunState:
    """Outcome of every guarded notebook cell, persisted after each cell so a failed or killed run
    keeps its history. Use it as `with state.cell("name"):` around a cell's body."""

    def __init__(
        self, path: str | Path | None = None, session_limit_s: float = KAGGLE_SESSION_LIMIT_S
    ):
        self.path = Path(path) if path else resolve_data_root() / RUN_STATE_FILE
        self.session_limit_s = session_limit_s
        self.started = time.time()
        self.cells: dict[str, dict] = {}
        self._save()

    def ok(self, name: str) -> bool:
        return self.cells.get(name, {}).get("status") == "ok"

    def require(self, *names: str) -> None:
        """Skips the current cell (cleanly) unless every named cell completed."""
        missing = [name for name in names if not self.ok(name)]
        if missing:
            raise CellSkipped(f"needs {missing} to have completed")

    def time_left(self) -> float:
        """Seconds a long job may still run before the session limit minus a safety margin."""
        return max(0.0, self.session_limit_s - SESSION_SAFETY_S - (time.time() - self.started))

    @contextlib.contextmanager
    def cell(self, name: str):
        started = time.time()
        status, detail = "ok", ""
        try:
            yield
        except CellSkipped as skip:
            status, detail = "skipped", str(skip)
            print(f"[skip] {name}: {detail}")
        except Exception as exc:  # noqa: BLE001 -- the point: nothing a cell raises may stop the run
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {name}: {detail}")
            traceback.print_exc()
        except BaseException:  # KeyboardInterrupt / SystemExit: record, then really stop
            self._record(name, "interrupted", "", started)
            raise
        self._record(name, status, detail, started)

    def _record(self, name: str, status: str, detail: str, started: float) -> None:
        self.cells[name] = {
            "status": status,
            "detail": detail,
            "seconds": round(time.time() - started, 1),
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cells": self.cells}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def print_summary(self, artifact_dirs: tuple[str | Path, ...] = ()) -> None:
        """Cell outcomes plus the files in `artifact_dirs`: what can be salvaged from this run."""
        print(f"{'cell':32s} {'status':12s} {'sec':>8s}  detail")
        for name, info in self.cells.items():
            print(f"{name:32s} {info['status']:12s} {info['seconds']:8.1f}  {info['detail'][:90]}")
        for directory in map(Path, artifact_dirs):
            files = sorted(p for p in directory.rglob("*") if p.is_file()) if directory.exists() else []
            print(f"\n{directory}: {len(files)} files")
            for path in files[:40]:
                print(f"  {path.relative_to(directory)}  ({path.stat().st_size / 1e6:.2f} MB)")


def ensure_utf8_output() -> None:
    """Makes stdout/stderr UTF-8 and non-raising. The training progress cards and tqdm bars print
    box-drawing characters; on Windows, with output going to a pipe or a file, the default cp1252
    encoding raises UnicodeEncodeError and kills a long run over a cosmetic character."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_logged(
    cmd: list[str],
    log_path: str | Path,
    timeout_s: float | None = None,
    cwd: str | Path | None = None,
    env: dict | None = None,
) -> int:
    """Runs a long job as a subprocess, mirroring its output to the notebook (progress bars
    throttled) and to `log_path`. Returns the exit code, `TIMEOUT_EXIT_CODE` if `timeout_s`
    elapsed (the job is terminated), and never raises on a failing job."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Children print Unicode (progress bars, cards): force UTF-8 mode so a Windows pipe cannot crash them.
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    proc = subprocess.Popen(  # noqa: S603 -- caller-supplied argv, never a shell string
        cmd, cwd=cwd, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", bufsize=1,
    )  # fmt: skip
    timed_out = threading.Event()

    def stop() -> None:
        timed_out.set()
        proc.terminate()

    watchdog = threading.Timer(timeout_s, stop) if timeout_s else None
    if watchdog:
        watchdog.start()
    last_progress = 0.0
    with open(log_path, "a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if PROGRESS_LINE.search(line) and time.time() - last_progress < 15:
                continue
            if PROGRESS_LINE.search(line):
                last_progress = time.time()
            print(line.rstrip(), flush=True)
    proc.wait()
    if watchdog:
        watchdog.cancel()
    return TIMEOUT_EXIT_CODE if timed_out.is_set() else proc.returncode


@dataclass
class PreflightReport:
    checks: list[tuple[str, str, str]] = field(default_factory=list)  # (name, ok|warn|fail, message)
    degraded: dict[str, str] = field(default_factory=dict)  # soft fallbacks callers should apply

    @property
    def ok(self) -> bool:
        return all(status != "fail" for _, status, _ in self.checks)

    def failures(self) -> list[str]:
        return [name for name, status, _ in self.checks if status == "fail"]

    def print(self) -> None:
        for name, status, message in self.checks:
            print(f"[{status:4s}] {name:18s} {message}")
        for name, note in self.degraded.items():
            print(f"[soft] {name:18s} {note}")


def preflight(
    need_gpu: bool = False,
    need_cdhit: bool = False,
    need_hub_write: bool = False,
    secrets: tuple[str, ...] = ("HF_TOKEN", "WANDB_API_KEY"),
    min_free_gb: float = 10.0,
    check_internet: bool = True,
) -> PreflightReport:
    """Reports every environment problem at once, before a long run starts. Secret VALUES are
    never printed. `degraded` lists soft fallbacks the caller should apply (e.g. W&B off)."""
    report = PreflightReport()
    add = lambda name, status, message: report.checks.append((name, status, message))  # noqa: E731

    add("platform", "ok", f"{detect_platform()} | python {sys.version.split()[0]}")

    # 1. GPU
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            add("gpu", "ok", f"{props.name}, {props.total_memory / 2**30:.0f} GB")
        else:
            add("gpu", "fail" if need_gpu else "warn", "no CUDA device" + ("" if need_gpu else " (CPU run)"))
    except ImportError:
        add("gpu", "fail" if need_gpu else "warn", "torch not installed")

    # 2. Disk
    root = resolve_data_root()
    free_gb = shutil.disk_usage(root if root.exists() else root.parent).free / 2**30
    add("disk", "ok" if free_gb >= min_free_gb else "warn", f"{free_gb:.0f} GB free at {root}")

    # 3. Internet
    if check_internet:
        try:
            urllib.request.urlopen("https://huggingface.co", timeout=8)
            add("internet", "ok", "huggingface.co reachable")
        except OSError as exc:
            add("internet", "fail", f"huggingface.co unreachable ({type(exc).__name__})")

    # 4. Secrets (names only)
    for name in secrets:
        if resolve_secret(name):
            add(f"secret {name}", "ok", "set")
        elif name == "HF_TOKEN":
            add(f"secret {name}", "fail" if need_hub_write else "warn", "missing")
        else:
            add(f"secret {name}", "warn", "missing")
            if name == "WANDB_API_KEY":
                report.degraded["wandb"] = "disabled: no WANDB_API_KEY (use --set wandb.enabled=false)"

    # 5. Hub write scope (role only; fine-grained tokens must be checked by the caller's first upload)
    if need_hub_write and (token := resolve_secret("HF_TOKEN")):
        try:
            from huggingface_hub import HfApi

            role = HfApi(token=token).whoami().get("auth", {}).get("accessToken", {}).get("role")
            status = {"write": "ok", "admin": "ok", "read": "fail"}.get(role, "warn")
            add("hub write scope", status, f"token role: {role}")
        except Exception as exc:  # noqa: BLE001 -- a failed lookup must not look like a failed token
            add("hub write scope", "warn", f"could not verify ({type(exc).__name__})")

    # 6. cd-hit
    if need_cdhit:
        missing = [b for b in ("cd-hit", "cd-hit-2d") if shutil.which(b) is None]
        add("cd-hit", "fail" if missing else "ok", f"missing {missing}" if missing else "on PATH")

    report.print()
    return report
