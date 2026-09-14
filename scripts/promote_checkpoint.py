"""Promote a shortlisted W&B run's checkpoint to the permanent Hugging Face Hub repo.

Tier 1 (W&B Artifacts) absorbs every ablation run cheaply. Only a handful of runs — the ones
that survive comparison across the 4-way architecture ablation — are worth a durable, sharable
copy. This script is that promotion step: it pulls a run's `best` artifact, pushes it to a
shared Hub model repo on its own revision, and writes a short model card recording where it
came from so the Hub copy stays traceable back to its full W&B metrics history.

Run by hand, after reviewing results — not part of the training loop.

Usage:
    uv run python scripts/promote_checkpoint.py \
        --wandb-run entity/ptm-sae/abc123 \
        --hub-repo mustafa-muhaimin/ptm-sae-checkpoints \
        --revision topk-k32-w4096
"""

import argparse
import json
import tempfile
from pathlib import Path

from ptm_sae.training.train import MODEL_CLASS_BY_TYPE


def promote_checkpoint(wandb_run_path: str, hub_repo: str, revision: str) -> None:
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)

    best_artifacts = [a for a in run.logged_artifacts() if "best" in a.aliases]
    if not best_artifacts:
        raise ValueError(f"No 'best' artifact found on run {wandb_run_path}")
    artifact = best_artifacts[-1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = Path(artifact.download(root=tmp_dir))
        sae_type = json.loads((artifact_dir / "config.json").read_text())["sae_type"]

        model = MODEL_CLASS_BY_TYPE[sae_type].from_pretrained(artifact_dir)
        model.push_to_hub(hub_repo, revision=revision, create_pr=False)

        card = (
            f"# {sae_type} SAE — promoted from W&B run `{wandb_run_path}`\n\n"
            f"- Source run: {run.url}\n"
            f"- sae_type: {sae_type}\n"
            f"- Config (from the run's W&B config): {json.dumps(dict(run.config), indent=2)}\n"
        )
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=hub_repo,
            revision=revision,
        )

    print(f"Promoted {wandb_run_path} -> {hub_repo}@{revision}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote a shortlisted W&B run's best checkpoint to the HF Hub"
    )
    parser.add_argument(
        "--wandb-run", type=str, required=True, help="W&B run path: entity/project/run_id"
    )
    parser.add_argument(
        "--hub-repo", type=str, required=True, help="Target HF Hub model repo, e.g. org/name"
    )
    parser.add_argument(
        "--revision", type=str, required=True, help="Hub revision/branch name for this run"
    )
    args = parser.parse_args()
    promote_checkpoint(args.wandb_run, args.hub_repo, args.revision)


if __name__ == "__main__":
    main()
