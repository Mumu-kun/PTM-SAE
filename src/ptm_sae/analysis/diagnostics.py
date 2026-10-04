"""Pass/fail verdicts and diagnostic plots for a training run's pulled W&B scalars.

Target ranges come from docs/research/sae-architectures-auxk-and-metrics.md and the Unified
Thesis Plan's hard exclusion floor. Checks are table-driven, split into `COMMON_CHECKS` (every
architecture) and `ARCH_CHECKS` (keyed by `sae_type`) so adding a new architecture later means
adding one dict entry here, not touching either notebook that calls into this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Status = Literal["pass", "fail", "n/a"]


@dataclass(frozen=True)
class CheckSpec:
    key: str  # column name in the pulled scalars DataFrame
    label: str  # human-readable name for tables/plots
    low: float | None  # target band lower bound, None = unbounded
    high: float | None  # target band upper bound, None = unbounded
    hard_floor: bool = (
        False  # True: failing this check means "exclude this run/arm" per the
    )
    # Unified Thesis Plan's hard floor, not just "outside the soft target band"
    log_scale: bool = False
    reference_value: float | None = (
        None  # e.g. the frozen-init threshold_std from the earlier
    )
    # failed run, drawn as a horizontal reference line rather than a pass/fail band


COMMON_CHECKS: list[CheckSpec] = [
    CheckSpec(
        "val/explained_variance", "Explained Variance", 0.85, 0.95, hard_floor=True
    ),
    CheckSpec("val/cosine_sim_mean", "Cosine Similarity (mean)", 0.92, None),
    CheckSpec("val/cosine_sim_p10", "Cosine Similarity (p10)", 0.85, None),
    CheckSpec("val/mse", "Reconstruction MSE", 0.010, 0.025),
    CheckSpec("train/decoder_pre_norm_mean", "Decoder Pre-Norm Mean", 0.90, 1.20),
    CheckSpec(
        "val/dead_latent_fraction", "Dead Latent Fraction", None, 0.05, hard_floor=True
    ),
    CheckSpec("val/alive_latent_jaccard", "Alive Latent Jaccard", 0.95, None),
]

# Architecture-specific checks, filled in only for the two architectures with real configs
# today. Add a new sae_type key here (with its own CheckSpec list) to extend to BatchTopK,
# Gated, Stratified, PolySAE, or Bespoke once those are actually trained.
ARCH_CHECKS: dict[str, list[CheckSpec]] = {
    "jumprelu": [
        CheckSpec(
            "train/threshold_std",
            "Threshold Std (log-scale)",
            None,
            None,
            log_scale=True,
            reference_value=3.5e-5,  # frozen-init value from the earlier starved-threshold run
        ),
        CheckSpec("train/threshold_mean", "Threshold Mean", None, None),
        # Decoder norm is deliberately unconstrained under the tanh+pre-act loss (folded into
        # the sparsity/pre-act terms as a trained per-latent importance term) — its drift away
        # from the ~1.0 starting point is the signal that this mechanism is actually active,
        # not a stability concern the way it would be under the old unit-norm-decoder loss.
        CheckSpec("train/decoder_norm_mean", "Decoder Norm Mean", None, None, reference_value=1.0),
    ],
    "topk": [
        CheckSpec("train/l0", "L0 (pinned near k)", None, None),
    ],
}

# Common checks that don't apply for a given architecture — jumprelu's decoder norm has no
# unit-norm target to measure drift from (see ARCH_CHECKS["jumprelu"]'s own decoder_norm_mean
# check instead), so the common band-based check would misreport a healthy run as failing.
_SKIP_COMMON_CHECKS: dict[str, set[str]] = {"jumprelu": {"train/decoder_pre_norm_mean"}}


def _checks_for(sae_type: str) -> list[CheckSpec]:
    skip = _SKIP_COMMON_CHECKS.get(sae_type, set())
    return [c for c in COMMON_CHECKS if c.key not in skip] + ARCH_CHECKS.get(sae_type, [])


@dataclass
class DiagnosticsResult:
    verdict: pd.DataFrame  # columns: metric, value, target, status
    recommendations: list[
        tuple[str, str, str]
    ]  # (observed condition, diagnosis, recommended change)


def _format_band(check: CheckSpec) -> str:
    if check.low is None and check.high is None:
        return "n/a"
    if check.low is None:
        return f"< {check.high:g}"
    if check.high is None:
        return f">= {check.low:g}"
    return f"{check.low:g} - {check.high:g}"


def _status(value: float, check: CheckSpec) -> Status:
    if check.low is None and check.high is None:
        return "n/a"
    if check.low is not None and value < check.low:
        return "fail"
    if check.high is not None and value > check.high:
        return "fail"
    return "pass"


def run_diagnostics(
    scalars: pd.DataFrame, sae_type: str, config: dict
) -> DiagnosticsResult:
    checks = _checks_for(sae_type)
    rows = []
    for check in checks:
        if check.key not in scalars.columns:
            continue
        value = scalars[check.key].dropna().iloc[-1]
        rows.append(
            {
                "metric": check.label,
                "value": round(value, 6),
                "target": _format_band(check),
                "status": _status(value, check),
            }
        )
    verdict = pd.DataFrame(rows)
    recommendations = _recommendations(scalars, sae_type, config)
    return DiagnosticsResult(verdict=verdict, recommendations=recommendations)


def _recommendations(
    scalars: pd.DataFrame, sae_type: str, config: dict
) -> list[tuple[str, str, str]]:
    if sae_type != "jumprelu":
        return [
            (
                "No jumprelu-specific recommendation rules defined for this sae_type.",
                "",
                "",
            )
        ]

    d_hidden = config["d_hidden"]
    l0_final = scalars["train/l0"].dropna().iloc[-1]
    l0_frac_final = l0_final / d_hidden
    threshold_std_max = scalars["train/threshold_std"].dropna().max()
    decoder_norm_final = (
        scalars["train/decoder_norm_mean"].dropna().iloc[-1]
        if "train/decoder_norm_mean" in scalars
        else None
    )
    ev_final = (
        scalars["val/explained_variance"].dropna().iloc[-1]
        if "val/explained_variance" in scalars
        else None
    )

    rules: list[tuple[bool, str, str, str]] = [
        (
            threshold_std_max < 10 * 3.5e-5,
            f"threshold_std peaked at {threshold_std_max:.2e}, still within ~1 order of "
            "magnitude of the earlier frozen-init failure (~3.5e-5).",
            "The threshold STE is still barely receiving gradient signal.",
            "Widen `bandwidth` further (the tanh+pre-act recipe's default is 2.0 — try raising "
            "it further) and/or raise `sparsity_coefficient`.",
        ),
        (
            decoder_norm_final is not None and abs(decoder_norm_final - 1.0) < 0.05,
            f"train/decoder_norm_mean ended at {decoder_norm_final:.3f}, barely moved from its "
            "~1.0 starting point.",
            "The tanh+pre-act loss's per-latent importance mechanism (feature magnitude = "
            "f_i(x) * ||W_dec,i||) isn't actually shaping training — decoder norms should drift "
            "away from 1.0 as the model learns which latents matter more.",
            "Raise `sparsity_coefficient` (the tanh term is what drives this) and confirm "
            "`remove_decoder_gradient_parallel_component_`/`normalize_decoder_` are actually "
            "being skipped for this run (they must be, for jumprelu).",
        ),
        (
            l0_frac_final > 0.5,
            f"train/l0 ended at {l0_final:.0f}/{d_hidden} ({l0_frac_final:.1%} of latents firing "
            "per token) — dense, not sparse.",
            "`sparsity_coefficient` is too weak relative to the MSE gradient once thresholds "
            "start moving: the model settles into an overcomplete, near-dense code instead of "
            "a sparse one.",
            f"Raise `sparsity_coefficient` substantially (current "
            f"{config.get('sparsity_coefficient')} — try 2-5x higher) and re-run; watch "
            "train/l0 actually trend down over the run, not just fail to trend down.",
        ),
        (
            l0_frac_final < 0.001,
            f"train/l0 collapsed to {l0_final:.0f}/{d_hidden} ({l0_frac_final:.1%}).",
            "`sparsity_coefficient` is too strong for the reconstruction objective to keep up.",
            "Lower `sparsity_coefficient` and re-run.",
        ),
        (
            ev_final is not None and ev_final < 0.90,
            f"val/explained_variance ended at {ev_final:.3f}, below the 0.90 hard floor.",
            "Reconstruction quality itself failed, independent of sparsity.",
            "Lower `sparsity_coefficient` and/or check the learning rate.",
        ),
    ]

    triggered = [(obs, diag, rec) for cond, obs, diag, rec in rules if cond]
    if not triggered:
        return [
            (
                "No known failure signature matched.",
                "Run may actually be healthy.",
                "Re-examine why it was flagged as bad.",
            )
        ]
    return triggered


def plot_checks(
    scalars: pd.DataFrame,
    sae_type: str,
    out_dir: Path,
    histogram: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[Path]:
    """One figure per check, saved as PNG under out_dir so nbconvert-executed runs leave
    viewable files behind (not just inline display output)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for check in _checks_for(sae_type):
        if check.key not in scalars.columns:
            continue
        saved.append(_plot_one(scalars, check, out_dir))

    if histogram is not None:
        saved.append(_plot_histogram(histogram, out_dir))

    return saved


def _plot_one(scalars: pd.DataFrame, check: CheckSpec, out_dir: Path) -> Path:
    series = scalars[check.key].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(series.index, series.values, color="#4C72B0")
    if check.log_scale:
        ax.set_yscale("log")
    if check.low is not None or check.high is not None:
        low = check.low if check.low is not None else ax.get_ylim()[0]
        high = check.high if check.high is not None else ax.get_ylim()[1]
        ax.axhspan(low, high, color="#55A868", alpha=0.15, label="target band")
    if check.reference_value is not None:
        ax.axhline(
            check.reference_value,
            color="#C44E52",
            linestyle="--",
            label="reference value",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(check.label)
    ax.set_title(check.label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{check.key.replace('/', '_')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_histogram(histogram: tuple[np.ndarray, np.ndarray], out_dir: Path) -> Path:
    counts, edges = histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#4C72B0")
    ax.set_xlabel("log10(firing density)")
    ax.set_ylabel("latent count")
    ax.set_title("Final-eval feature density histogram")
    fig.tight_layout()
    path = out_dir / "feature_density_histogram.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
