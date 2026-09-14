"""Delete stale, unshortlisted W&B runs (and their logged artifacts) from the ablation project.

Most runs in a sweep are never worth keeping once a winner is picked. Deleting a run also
deletes everything it logged via wandb.Artifact (see train.py's _log_checkpoint_artifact), so
there is no separate artifact-cleanup step.

This never runs automatically and never touches a run you haven't tagged. Tag a promising run
`shortlist` (W&B UI, or `wandb_run.tags = [*wandb_run.tags, "shortlist"]` from a notebook cell)
as soon as its numbers look good, before deciding whether to promote it to the Hub — the tag is
what protects it here, independent of promotion.

Usage:
    uv run python scripts/cleanup_wandb_runs.py --project ptm-sae            # dry run, default
    uv run python scripts/cleanup_wandb_runs.py --project ptm-sae --yes      # actually delete
    uv run python scripts/cleanup_wandb_runs.py --project ptm-sae --older-than-days 30 --yes
"""

import argparse
from datetime import UTC, datetime, timedelta


def cleanup_wandb_runs(project: str, older_than_days: int, dry_run: bool) -> None:
    import wandb

    api = wandb.Api()
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

    for run in api.runs(project):
        created_at = datetime.fromisoformat(run.created_at).replace(tzinfo=UTC)
        if created_at >= cutoff:
            continue
        if "shortlist" in run.tags:
            continue

        action = "Would delete" if dry_run else "Deleting"
        print(f"{action}: {run.path} '{run.name}' (created {created_at.date()}, tags={run.tags})")
        if not dry_run:
            run.delete()

    if dry_run:
        print("\nDry run — no runs deleted. Pass --yes to actually delete the runs listed above.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete stale, unshortlisted W&B runs and their logged artifacts"
    )
    parser.add_argument("--project", type=str, required=True, help="W&B project, e.g. ptm-sae")
    parser.add_argument(
        "--older-than-days", type=int, default=14, help="Only consider runs older than this"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Actually delete (default is a dry run that only prints)"
    )
    args = parser.parse_args()
    cleanup_wandb_runs(args.project, args.older_than_days, dry_run=not args.yes)


if __name__ == "__main__":
    main()
