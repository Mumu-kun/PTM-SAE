"""Verify the notebooks end to end in smoke mode, and only when the code they depend on has changed.

A notebook can break without any test failing (a changed CLI flag, a moved path, a package pinned
newer than the platform's). So after every change to what the notebooks run, execute them in smoke
mode on the platform they target and check the run summary:

    uv run --extra verify python scripts/verify_notebooks.py status
    uv run --extra verify python scripts/verify_notebooks.py run --target local  --if-changed
    uv run --extra verify python scripts/verify_notebooks.py run --target kaggle --if-changed

Targets: `local` runs the notebooks headlessly with nbconvert; `kaggle` syncs the working tree to a live
Kaggle kernel through `kgz` (unofficial; reviewed version pinned in the `verify` extra), restarts it and executes the cells in order, stopping at the
first uncaught error exactly as a Kaggle "Save & Run All" would. The kernel URL is a credential: it is
read from a file (default ~/.kaggle_kernel_url, override with --url-file) and never printed.

A target counts as verified for the current fingerprint of the watched paths; any change to them
makes `status` report it stale and `--if-changed` run again. State lives in .verify_state.json
(untracked).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_FILE = REPO / ".verify_state.json"
LOG_DIR = REPO / ".verify_logs"  # logs of failed runs are kept here (untracked)
WATCHED = ("notebooks", "src/ptm_sae", "configs", "pyproject.toml")  # what a notebook run depends on
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
NOTEBOOKS = {"train": "notebooks/train_sae.ipynb", "extract": "notebooks/kaggle_pipeline.ipynb"}
# Cells that must have completed (not merely skipped) in a smoke run.
REQUIRED_OK = {
    "train": ["install", "preflight", "config", "train", "readback"],
    "extract": ["install", "preflight", "extraction", "readback"],
}
KAGGLE_ZIP_INCLUDE = ("src/ptm_sae", "configs", "tests", "pyproject.toml", "README.md", "data/sample.fasta")
NL = chr(10)


def watched_files(repo: Path = REPO) -> list[Path]:
    files = []
    for entry in WATCHED:
        path = repo / entry
        found = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        files += [p for p in found if not SKIP_PARTS & set(p.parts) and p.suffix != ".pyc"]
    return files


def watched_hash(repo: Path = REPO) -> str:
    """Fingerprint of every file a notebook run depends on (path + content)."""
    digest = hashlib.sha256()
    for path in watched_files(repo):
        digest.update(path.relative_to(repo).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_state(path: Path = STATE_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def is_verified(state: dict, target: str, current_hash: str) -> bool:
    entry = state.get(target, {})
    return entry.get("ok") is True and entry.get("hash") == current_hash


def evaluate(key: str, cells: dict) -> list[str]:
    """Problems with a smoke run's `run_state.json` cells: any failed/interrupted cell, or a required
    cell that did not complete."""
    problems = [f"{name}: {info['status']} ({info.get('detail', '')[:100]})" for name, info in cells.items() if info["status"] in ("failed", "interrupted")]
    problems += [f"{name} did not complete (status {cells.get(name, {}).get('status', 'missing')})" for name in REQUIRED_OK[key] if cells.get(name, {}).get("status") != "ok"]
    return problems


def run_local(key: str) -> tuple[dict, list[str]]:
    """Headless nbconvert run in smoke mode against a throwaway data root."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        env = {"PTM_SAE_MODE": "smoke", "PTM_SAE_PULL": "0", "PTM_SAE_DATA_ROOT": str(scratch / "dataroot"), "PYTHONUTF8": "1"}
        notebook = REPO / NOTEBOOKS[key]
        done = subprocess.run(  # noqa: S603 -- our own notebook, argv list
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=1500",
             "--output-dir", str(scratch), "--output", "run.ipynb", str(notebook)],
            cwd=notebook.parent, capture_output=True, text=True, env={**os.environ, **env},
        )  # fmt: skip
        if done.returncode != 0:
            return {}, [f"nbconvert failed (an uncaught cell error): {done.stderr.strip().splitlines()[-1][:200] if done.stderr.strip() else 'no output'}"]
        state_file = scratch / "dataroot" / "run_state.json"
        if not state_file.exists():
            return {}, ["no run_state.json written: the notebook never reached its guarded cells"]
        cells = json.loads(state_file.read_text())["cells"]
        problems = evaluate(key, cells)
        if problems:  # keep the evidence: the scratch directory disappears when this block ends
            keep = LOG_DIR / f"local-{key}"
            shutil.rmtree(keep, ignore_errors=True)
            if (scratch / "dataroot" / "logs").exists():
                shutil.copytree(scratch / "dataroot" / "logs", keep)
            for log in sorted(keep.glob("*.log")):
                print(f"--- tail of {log.name} (saved in {keep}) ---")
                print(NL.join(log.read_text(errors="replace").replace(chr(13), NL).splitlines()[-12:]))
    return cells, problems


def sync_to_kernel(kernel, base_url: str) -> int:
    from kgz.file_ops import upload_file

    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "ptm_sae_src.zip"
        count = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in KAGGLE_ZIP_INCLUDE:
                path = REPO / item
                for file in [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()):
                    if SKIP_PARTS & set(file.parts) or file.suffix == ".pyc":
                        continue
                    zf.write(file, file.relative_to(REPO).as_posix())
                    count += 1
        upload_file(base_url, str(archive), "ptm_sae_src.zip")
    kernel.execute(
        "import os, shutil, zipfile" + NL + "shutil.rmtree('/kaggle/working/PTM-SAE', ignore_errors=True)" + NL
        + "zipfile.ZipFile('/kaggle/working/ptm_sae_src.zip').extractall('/kaggle/working/PTM-SAE')",
        stream=False,
    )
    return count


def run_kaggle(key: str, url_file: Path, restart: bool = True) -> tuple[dict, list[str]]:
    """Sync the tree to the live kernel, restart it (a fresh process, like a new container), execute the
    notebook's code cells in order in smoke mode and read the run state back."""
    try:
        from kgz import Kernel
    except ImportError:
        return {}, ["kgz is not installed: run with `uv run --extra verify ...`"]
    if not url_file.exists():
        return {}, [f"kernel URL file {url_file} not found (copy the Kaggle kernel URL into it; it is a credential)"]

    cells_src = [
        "".join(c["source"])
        for c in json.loads((REPO / NOTEBOOKS[key]).read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]
    token_like = re.compile(r"[A-Za-z0-9_-]{40,}")
    try:
        kernel = Kernel(url_file.read_text().strip(), name="verify")
        print(f"[kaggle] synced {sync_to_kernel(kernel, kernel.base_url)} files")
        if restart:
            kernel.restart()
        kernel.execute(
            "import os" + NL + "os.environ.update({'PTM_SAE_MODE': 'smoke', 'PTM_SAE_PULL': '0'})" + NL + "os.chdir('/kaggle/working/PTM-SAE')",
            stream=False,
        )
        for index, source in enumerate(cells_src, start=1):
            result = kernel.execute(source, timeout=1800, stream=False)
            print(f"[kaggle] {key} cell {index}/{len(cells_src)}: {'ok' if result.success else 'UNCAUGHT ' + str(result.error_name)}")
            if not result.success:
                detail = token_like.sub("<redacted>", str(result.error_value))[:300]
                return {}, [f"uncaught error in cell {index} ({result.error_name}): {detail}"]
        reply = kernel.execute(
            "import json" + NL + "from ptm_sae import runtime" + NL + "print('RUNSTATE=' + json.dumps(json.load(open(runtime.resolve_data_root() / 'run_state.json'))['cells']))",
            stream=False,
        )
        kernel.close()
    except Exception as exc:  # noqa: BLE001 -- messages can embed the kernel URL
        return {}, [f"kgz failed: {type(exc).__name__} (message withheld: may contain the kernel URL)"]
    match = re.search(r"RUNSTATE=(.*)", reply.stdout or "")
    if not match:
        return {}, ["could not read run_state.json back from the kernel"]
    cells = json.loads(match.group(1))
    problems = evaluate(key, cells)
    if problems:
        tails = kernel_logs.stdout if (kernel_logs := _kernel_log_tails(url_file)) else ""
        print(tails)
    return cells, problems


def _kernel_log_tails(url_file: Path):
    """Last lines of every log the notebook wrote on the kernel (best effort, token-like strings masked)."""
    from kgz import Kernel

    try:
        kernel = Kernel(url_file.read_text().strip(), name="verify-logs")
        reply = kernel.execute(
            "from pathlib import Path" + NL + "from ptm_sae import runtime" + NL
            + "for log in sorted((runtime.resolve_data_root() / 'logs').glob('*.log')):" + NL
            + "    print('--- tail of', log.name, '---')" + NL
            + "    print(chr(10).join(log.read_text(errors='replace').replace(chr(13), chr(10)).splitlines()[-12:]))",
            stream=False,
        )
        kernel.close()
    except Exception:  # noqa: BLE001 -- evidence is best effort; the URL must not leak
        return None
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="is each target verified for the current code?")
    run = sub.add_parser("run", help="run the smoke verification")
    run.add_argument("--target", choices=["local", "kaggle"], required=True)
    run.add_argument("--notebook", choices=[*NOTEBOOKS, "all"], default="all")
    run.add_argument("--if-changed", action="store_true", help="skip when this target is already verified for the current code")
    run.add_argument("--url-file", type=Path, default=Path.home() / ".kaggle_kernel_url")
    run.add_argument("--no-restart", action="store_true", help="kaggle: do not restart the kernel first")
    args = parser.parse_args()

    state, current = load_state(), watched_hash()
    if args.command == "status":
        for target in ("local", "kaggle"):
            entry = state.get(target, {})
            fresh = is_verified(state, target, current)
            print(f"{target:7s} {'VERIFIED for current code' if fresh else 'STALE (needs verification)':30s} last: {entry.get('time', 'never')} {'ok' if entry.get('ok') else ''}")
        return

    if args.if_changed and is_verified(state, args.target, current):
        print(f"[{args.target}] already verified for the current code: nothing to do")
        return
    keys = list(NOTEBOOKS) if args.notebook == "all" else [args.notebook]
    all_problems: dict[str, list[str]] = {}
    not_run: list[str] = []
    for key in keys:
        print(f"[{args.target}] verifying {NOTEBOOKS[key]} (smoke) ...")
        cells, problems = run_local(key) if args.target == "local" else run_kaggle(key, args.url_file, not args.no_restart)
        skipped = [p for p in problems if p.startswith("skipped:")]
        if skipped and not cells:
            not_run.append(key)
        all_problems[key] = [] if skipped and not cells else problems
        for name, info in cells.items():
            print(f"    {name:20s} {info['status']:11s} {info.get('detail', '')[:80]}")
        print(f"[{args.target}] {key}: {'OK' if not all_problems[key] else 'PROBLEMS'}" + (f" ({skipped[0]})" if skipped else ""))
        for problem in all_problems[key] if not skipped else []:
            print("    -", problem)

    ok = not any(all_problems.values())
    state[args.target] = {"hash": current, "ok": ok, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "notebooks": [k for k in keys if k not in not_run], "not_run": not_run}
    STATE_FILE.write_text(json.dumps(state, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
