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
import tempfile
from pathlib import Path

MODE_DEFAULT = 'os.environ.get("PTM_SAE_MODE", "real")'
PHASES_DEFAULT = 'os.environ.get("PTM_SAE_PHASES", "corpus")'


def inject_mode(notebook: dict, mode: str, phases: str | None = None) -> dict:
    """Copy of `notebook` whose PTM_SAE_MODE default is `mode` (smoke | real) and, when given, whose
    PTM_SAE_PHASES default is `phases` (corpus | extraction | both)."""
    if mode not in ("smoke", "real"):
        raise ValueError(f"mode must be 'smoke' or 'real', not {mode!r}")
    if phases not in (None, "corpus", "extraction", "both"):
        raise ValueError(f"phases must be 'corpus', 'extraction' or 'both', not {phases!r}")
    patched = json.loads(json.dumps(notebook))
    replaced = 0
    for cell in patched["cells"]:
        source = "".join(cell["source"])
        if MODE_DEFAULT in source:
            replaced += 1
            source = source.replace(MODE_DEFAULT, f'os.environ.get("PTM_SAE_MODE", "{mode}")')
            if phases:
                source = source.replace(PHASES_DEFAULT, f'os.environ.get("PTM_SAE_PHASES", "{phases}")')
            lines = source.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if not replaced:
        raise ValueError("notebook has no PTM_SAE_MODE parameter to set")
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
    push.add_argument("--phases", choices=["corpus", "extraction", "both"], help="corpus notebook only; default corpus (CPU); extraction needs --gpu")
    push.add_argument("--gpu", action="store_true", help="GPU accelerator (default: CPU only, saves the weekly GPU quota)")
    for name in ("status", "output"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--slug", required=True)
    sub.choices["output"].add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    api, username = _api()
    kernel = f"{username}/{args.slug}"

    if args.command == "push":
        notebook = inject_mode(json.loads(args.notebook.read_text(encoding="utf-8")), args.mode, args.phases)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / args.notebook.name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
            metadata = build_metadata(username, args.slug, args.notebook.name, args.gpu)
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(api.kernels_push(str(folder)))
        print(f"pushed {kernel} (mode={args.mode}, phases={args.phases}, gpu={args.gpu}); status: python scripts/kaggle_run.py status --slug {args.slug}")
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
