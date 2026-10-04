"""Programmatic W&B workspace layout for live monitoring while a run is in progress.

Training (`ptm_sae.training.train`) blocks the notebook kernel for its whole duration, so
"monitor while training" has to happen in the W&B web UI in a separate browser tab — this
groups that dashboard's panels into the same sections the post-hoc diagnostics use (see
`ptm_sae.analysis.diagnostics`), instead of leaving them in W&B's default flat auto-layout.

Run once (or whenever the section/panel list below changes) via:
    uv run python -m ptm_sae.analysis.wandb_workspace --project ptm-sae
then open the printed URL — it's a saved view, so it persists across runs.
"""

import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

# (section title, [(metric, log_y, range_y), ...]) — range_y is a W&B axis-range hint standing
# in for a target band; W&B panels don't support a shaded reference band natively.
_SECTIONS: list[
    tuple[str, list[tuple[str, bool, tuple[float | None, float | None]]]]
] = [
    (
        "Sparsity & Threshold",
        [
            ("train/l0", False, (None, None)),
            ("train/threshold_mean", False, (None, None)),
            ("train/threshold_std", True, (None, None)),
        ],
    ),
    (
        "Reconstruction Quality",
        [
            ("val/explained_variance", False, (0.80, 1.0)),
            ("val/cosine_sim_mean", False, (0.80, 1.0)),
            ("val/cosine_sim_p10", False, (0.70, 1.0)),
            ("val/mse", False, (None, None)),
        ],
    ),
    (
        "Health",
        [
            ("val/dead_latent_fraction", False, (0.0, 0.6)),
            ("val/alive_latent_jaccard", False, (0.0, 1.0)),
            ("train/decoder_pre_norm_mean", False, (0.7, 1.4)),
        ],
    ),
    (
        "Throughput",
        [
            ("train/tokens_per_sec", False, (None, None)),
            ("train/eta_seconds", False, (None, None)),
        ],
    ),
]


def build_diagnostic_workspace(
    entity: str, project: str, name: str = "SAE diagnostics"
) -> str:
    sections = [
        ws.Section(
            name=title,
            panels=[
                wr.LinePlot(title=metric, y=[metric], log_y=log_y, range_y=range_y)
                for metric, log_y, range_y in metrics
            ],
        )
        for title, metrics in _SECTIONS
    ]
    workspace = ws.Workspace(
        entity=entity, project=project, name=name, sections=sections
    )
    workspace.save()
    # workspace.url joins entity/project with os.path.join, which emits backslashes on Windows
    # and produces a broken URL — rebuild it with an explicit forward slash instead.
    url = workspace.url
    return url.replace(f"{entity}\\{project}", f"{entity}/{project}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="(Re)create the SAE training diagnostic W&B workspace"
    )
    parser.add_argument(
        "--project", type=str, required=True, help="W&B project, e.g. ptm-sae"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity; defaults to your default entity",
    )
    args = parser.parse_args()

    import wandb

    entity = args.entity or wandb.Api().default_entity
    url = build_diagnostic_workspace(entity, args.project)
    print(f"Workspace saved: {url}")


if __name__ == "__main__":
    main()
