"""Push a notebook to Kaggle as a committed (non-interactive "Save & Run All") run, watch it, fetch it.

Verify the notebook first (smoke mode locally / on the GPU box, then on a Kaggle kernel), then:

    uv run --extra kaggle python scripts/kaggle_run.py push notebooks/kaggle_pipeline.ipynb \
        --slug ptm-sae-corpus-build --mode real            # CPU-only by default; --gpu for a GPU run
    uv run --extra kaggle python scripts/kaggle_run.py status --slug ptm-sae-corpus-build
    uv run --extra kaggle python scripts/kaggle_run.py output --slug ptm-sae-corpus-build --out runs/corpus

A pushed notebook cannot receive environment variables, so `--mode` is written into the pushed copy
(it replaces the default of PTM_SAE_MODE). Secrets (HF_TOKEN, WANDB_API_KEY, GH_TOKEN) are not part
of the push: attach them to the notebook once in the Kaggle UI (Add-ons -> Secrets); the notebook's
preflight cell reports which are missing, by name only. Credentials come from ~/.kaggle/kaggle.json.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path


def _set_default(cells: list[dict], name: str, value: str) -> int:
    """Rewrites the default of every `os.environ.get("NAME", "...")` in `cells`; returns how many it changed."""
    pattern = re.compile(rf'os\.environ\.get\("{re.escape(name)}",\s*"[^"]*"\)')
    replacement = f'os.environ.get("{name}", "{value}")'
    changed = 0
    for cell in cells:
        source, count = pattern.subn(lambda _match: replacement, "".join(cell["source"]))
        if count:
            changed += count
            lines = source.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return changed


def inject_mode(
    notebook: dict, mode: str, params: dict[str, str] | None = None
) -> dict:
    """Copy of `notebook` whose PTM_SAE_MODE default is `mode` (smoke | real). `params` sets the default of
    any other `os.environ.get("NAME", "...")` parameter, e.g. {"PTM_SAE_SWEEP_SPEC": "sweeps/topk_sweep.yaml"}."""
    if mode not in ("smoke", "real"):
        raise ValueError(f"mode must be 'smoke' or 'real', not {mode!r}")
    patched = json.loads(json.dumps(notebook))
    defaults = {"PTM_SAE_MODE": mode, **(params or {})}
    for name, value in defaults.items():
        if not _set_default(patched["cells"], name, value):
            raise ValueError(f"notebook has no {name} parameter to set")
    return patched


def build_metadata(username: str, slug: str, notebook_file: str, gpu: bool = False) -> dict:
    """kernel-metadata.json for a private notebook with internet on (the data steps need it)."""
    return {
        "id": f"{username}/{slug}",
        "title": slug.replace("-", " "),  # Kaggle derives the slug from the title: they must match
        "code_file": notebook_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": str(gpu).lower(),
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api, json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    push = sub.add_parser("push", help="push a notebook and start a committed run")
    push.add_argument("notebook", type=Path)
    push.add_argument("--slug", required=True, help="kernel slug, e.g. ptm-sae-corpus-build")
    push.add_argument("--mode", choices=["smoke", "real"], default="real")
    push.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="set another notebook parameter default, e.g. PTM_SAE_QUICK=1 (repeatable)")
    push.add_argument("--gpu", action="store_true", help="GPU accelerator (default: CPU only, saves the weekly GPU quota)")
    for name in ("status", "output"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--slug", required=True)
    sub.choices["output"].add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    api, username = _api()
    kernel = f"{username}/{args.slug}"

    if args.command == "push":
        params = dict(item.split("=", 1) for item in args.param)
        notebook = inject_mode(json.loads(args.notebook.read_text(encoding="utf-8")), args.mode, params)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / args.notebook.name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
            metadata = build_metadata(username, args.slug, args.notebook.name, args.gpu)
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(api.kernels_push(str(folder)))
        print(f"pushed {kernel} (mode={args.mode}, gpu={args.gpu}); status: python scripts/kaggle_run.py status --slug {args.slug}")
    elif args.command == "status":
        status = api.kernels_status(kernel)
        print(f"{kernel}: {status.status} {getattr(status, 'failure_message', None) or ''}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        api.kernels_output(kernel, path=str(args.out))
        files = sorted(p for p in args.out.rglob("*") if p.is_file())
        print(f"downloaded {len(files)} files to {args.out}")
        for path in files[:40]:
            print(f"  {path.relative_to(args.out)}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
