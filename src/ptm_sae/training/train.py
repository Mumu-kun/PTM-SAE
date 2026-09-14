"""Training loop for baseline TopK / JumpReLU / BatchTopK / Gated SAEs.

Streams discovery_train via ActivationPartitionDataset, trains with AdamW under each
architecture's literature-matched recipe (TopK/BatchTopK: constant LR, Gao et al. 2024/
Bussmann 2024; JumpReLU: warmup+cosine LR plus a separately warmed-up L0 coefficient,
Rajamanoharan et al. 2024; Gated: constant LR, Rajamanoharan et al. 2024a), maintains the
unit-norm decoder constraint each step, tracks a token-windowed dead-latent census, and
periodically evaluates on discovery_val with local-only checkpointing (`latest` + `best`,
matching a run's own `checkpoint_dir`).
"""

import argparse
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from ptm_sae.extraction.progress import PipelineProgressManager
from ptm_sae.models import (
    BatchTopKSAEConfig,
    BatchTopKSAEModel,
    GatedSAEConfig,
    GatedSAEModel,
    JumpReLUSAEConfig,
    JumpReLUSAEModel,
    SAEOutput,
    TopKSAEConfig,
    TopKSAEModel,
)
from ptm_sae.training.collapse_check import (
    load_discovery_val_labels,
    run_collapse_check,
)
from ptm_sae.training.config import SAETrainingConfig
from ptm_sae.training.dataset import build_partition_dataloader

SAEModel = TopKSAEModel | JumpReLUSAEModel | BatchTopKSAEModel | GatedSAEModel

# Architectures whose forward() accepts dead_latent_mask (AuxK support) — everything else
# (JumpReLU, Gated) doesn't take that keyword at all.
_AUXK_ARCHITECTURES = ("topk", "batchtopk")

MODEL_CLASS_BY_TYPE = {
    "topk": TopKSAEModel,
    "batchtopk": BatchTopKSAEModel,
    "gated": GatedSAEModel,
    "jumprelu": JumpReLUSAEModel,
}


def _build_model(config: SAETrainingConfig) -> SAEModel:
    if config.sae_type in _AUXK_ARCHITECTURES:
        # None means "let TopKSAEConfig/BatchTopKSAEConfig's own published default apply" —
        # off for TopK, on at 1/32 for BatchTopK. Only override when the user set one.
        shared_kwargs = {}
        if config.auxk_coefficient is not None:
            shared_kwargs["auxk_coefficient"] = config.auxk_coefficient
        if config.k_aux is not None:
            shared_kwargs["k_aux"] = config.k_aux

        if config.sae_type == "topk":
            return TopKSAEModel(
                TopKSAEConfig(
                    d_in=config.d_in, d_hidden=config.d_hidden, k=config.k, **shared_kwargs
                )
            )
        return BatchTopKSAEModel(
            BatchTopKSAEConfig(
                d_in=config.d_in,
                d_hidden=config.d_hidden,
                k=config.k,
                threshold_ema_decay=config.threshold_ema_decay,
                **shared_kwargs,
            )
        )

    if config.sae_type == "gated":
        return GatedSAEModel(GatedSAEConfig(d_in=config.d_in, d_hidden=config.d_hidden))

    return JumpReLUSAEModel(
        JumpReLUSAEConfig(
            d_in=config.d_in,
            d_hidden=config.d_hidden,
            bandwidth=config.bandwidth,
            init_threshold=config.init_threshold,
        )
    )


def _save_resume_state(
    latest_dir: Path,
    step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tokens_since_fired: torch.Tensor,
    best_val_mse: float,
    wandb_run_id: str | None,
) -> None:
    """Written inside checkpoint_dir/latest/ (not the checkpoint_dir root) so a single
    wandb.Artifact.add_dir(latest/) call bundles the whole resume bundle alongside the HF
    weights in one upload. checkpoint_dir/best/ never gets this file — it stays pure HF
    weights, since that's the one that gets promoted to the Hub or shared, and optimizer
    moment buffers have no business riding along with a result checkpoint."""
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "tokens_since_fired": tokens_since_fired,
            "best_val_mse": best_val_mse,
            "wandb_run_id": wandb_run_id,
        },
        latest_dir / "resume_state.pt",
    )


def _load_resume_state(latest_dir: Path, device: torch.device) -> dict:
    return torch.load(latest_dir / "resume_state.pt", map_location=device, weights_only=False)


def _log_checkpoint_artifact(wandb_run, checkpoint_dir: Path, sae_type: str, alias: str) -> None:
    """Logs a checkpoint dir as a W&B Artifact. Artifacts are natively attached to the run
    that created them (shown on the run's page), which is what makes a checkpoint "paired
    with its run" without any manifest file of our own to maintain."""
    import wandb

    artifact = wandb.Artifact(name=f"sae-{sae_type}", type="model")
    artifact.add_dir(str(checkpoint_dir))
    wandb_run.log_artifact(artifact, aliases=[alias])


def _forward(
    config: SAETrainingConfig, model: SAEModel, batch: torch.Tensor, dead_latent_mask: torch.Tensor
) -> SAEOutput:
    """Dispatches to the right forward() signature — only TopK/BatchTopK accept
    dead_latent_mask at all; passing it to JumpReLU/Gated would raise a TypeError."""
    if config.sae_type in _AUXK_ARCHITECTURES:
        return model(batch, dead_latent_mask=dead_latent_mask)
    return model(batch)


def _total_loss(
    config: SAETrainingConfig, model: SAEModel, out: SAEOutput, step: int
) -> torch.Tensor:
    if config.sae_type in _AUXK_ARCHITECTURES:
        loss = out.mse_loss
        if out.aux_loss is not None:
            # The coefficient lives on the model's own config (TopKSAEConfig/BatchTopKSAEConfig)
            # since that's what actually decided whether aux_loss was computed at all — not a
            # second copy of the same number kept on SAETrainingConfig.
            loss = loss + model.config.auxk_coefficient * out.aux_loss
        return loss

    if config.sae_type == "jumprelu":
        # L0 coefficient is linearly warmed up over its own schedule, separate from the LR
        # warmup (Rajamanoharan et al., 2024 warm it up over 10k steps / 40M tokens to avoid
        # collapsing every latent to zero before the SAE has learned anything reconstructive).
        ramped_l0_coef = config.l0_coefficient * min(1.0, step / max(1, config.l0_warmup_steps))
        return out.mse_loss + ramped_l0_coef * out.l0

    # gated: Eq. 8 (Rajamanoharan et al., 2024a) — main MSE + lambda*L1(gate) + aux reconstruction,
    # the auxiliary term added at weight 1 directly, no separate coefficient in the paper's loss.
    return out.mse_loss + config.l1_coefficient * out.l1_loss + out.aux_loss


class _StreamingEvalStats:
    """Single-pass, per-dimension running statistics for MSE / explained variance / L0, plus
    two cheap generic interpretability diagnostics that need no PTM labels: a per-latent
    firing-density histogram (dead-latent census only tells you the tail; this shows the whole
    distribution — a spike of always-on latents is as much a red flag as a dead tail) and
    reconstruction cosine similarity (a scale-invariant complement to MSE)."""

    def __init__(self, d_in: int, d_hidden: int, density_histogram_bins: int, device: torch.device):
        self.sum_x = torch.zeros(d_in, device=device)
        self.sum_x2 = torch.zeros(d_in, device=device)
        self.sum_sq_err = torch.tensor(0.0, device=device)
        self.sum_l0 = torch.tensor(0.0, device=device)
        self.n_rows = 0
        self.fire_count = torch.zeros(d_hidden, device=device)
        self.cosine_sims: list[torch.Tensor] = []
        self.density_histogram_bins = density_histogram_bins

    def update(
        self, x: torch.Tensor, reconstruction: torch.Tensor, l0: torch.Tensor, latents: torch.Tensor
    ) -> None:
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += (x**2).sum(dim=0)
        self.sum_sq_err += (x - reconstruction).pow(2).sum()
        self.sum_l0 += l0 * x.shape[0]
        self.n_rows += x.shape[0]
        self.fire_count += latents.gt(0).sum(dim=0).float()
        self.cosine_sims.append(torch.nn.functional.cosine_similarity(x, reconstruction, dim=-1))

    def finalize(self) -> tuple[dict[str, float], torch.Tensor]:
        """Returns (scalar metrics, alive-latent boolean mask). The mask is handed back
        separately (not logged as a metric itself) so the caller can track its Jaccard overlap
        with the previous eval's mask — a stability signal, not a per-eval scalar."""
        n = max(1, self.n_rows)
        mean_x = self.sum_x / n
        var_x_per_dim = (self.sum_x2 / n) - mean_x**2
        ss_tot = (var_x_per_dim.clamp_min(0) * n).sum()
        # sum_sq_err is summed over rows AND feature dims, so divide by both to match the
        # per-element MSE the training loop logs (`reconstruction_loss`'s `.pow(2).mean()`).
        mse = (self.sum_sq_err / (n * self.sum_x.shape[0])).item()
        explained_variance = 1.0 - (self.sum_sq_err / ss_tot.clamp_min(1e-8)).item()

        cosine = torch.cat(self.cosine_sims)
        density = self.fire_count / n
        alive_mask = density > 0

        alive_density = density[alive_mask]
        if alive_density.numel() > 0:
            # Log-spaced buckets over (0, 1] firing-fraction — density spans orders of
            # magnitude (a latent firing on 0.01% of tokens vs. 50% of them), so linear bins
            # would flatten the whole distribution into the first bucket.
            hist = torch.histc(
                torch.log10(alive_density.clamp_min(1e-12)),
                bins=self.density_histogram_bins,
                min=-12.0,
                max=0.0,
            )
        else:
            hist = torch.zeros(self.density_histogram_bins)

        metrics = {
            "mse": mse,
            "explained_variance": explained_variance,
            "mean_l0": (self.sum_l0 / n).item(),
            "cosine_sim_mean": cosine.mean().item(),
            "cosine_sim_p10": torch.quantile(cosine, 0.10).item(),
            "feature_density_histogram": hist.tolist(),
        }
        return metrics, alive_mask


@torch.no_grad()
def _evaluate(
    model: SAEModel,
    val_loader,
    config: SAETrainingConfig,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    """Runs one deterministic pass over discovery_val. `val_loader` is built once by the
    caller and reused across every eval checkpoint — discovery_val never needs reshuffling.
    Calling model(batch) uniformly (no dead_latent_mask) is safe for every architecture: eval
    mode always skips AuxK regardless of the mask argument (see TopKSAEModel.forward)."""
    model.eval()
    stats = _StreamingEvalStats(config.d_in, config.d_hidden, config.density_histogram_bins, device)
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch)
        stats.update(batch, out.reconstruction, out.l0, out.latents)
    model.train()
    return stats.finalize()


def run_sae_training(
    config: SAETrainingConfig,
    progress_manager: PipelineProgressManager | None = None,
) -> dict:
    pm = progress_manager or PipelineProgressManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if config.dtype == "bf16" else None

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resuming reconstructs the model from a PREVIOUS run's checkpoint_dir/latest/ (weights +
    # optimizer/scheduler/step/dead-latent-census/wandb_run_id) rather than building fresh.
    # No RNG state needs restoring: ActivationPartitionDataset.set_epoch() reseeds shuffling
    # purely from (seed, epoch, worker_id), and none of the four architectures use dropout.
    resume_state: dict | None = None
    if config.resume_from is not None:
        resume_latest_dir = Path(config.resume_from) / "latest"
        model = MODEL_CLASS_BY_TYPE[config.sae_type].from_pretrained(resume_latest_dir).to(device)
        resume_state = _load_resume_state(resume_latest_dir, device)
    else:
        model = _build_model(config).to(device)

    wandb_run = None
    if config.wandb.enabled:
        import wandb

        resumed_run_id = resume_state["wandb_run_id"] if resume_state is not None else None
        wandb_run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            config=config.model_dump(),
            tags=config.wandb.tags,
            id=resumed_run_id,
            resume="must" if resumed_run_id is not None else None,
        )

    collapse_labels = None
    if config.enable_collapse_check:
        collapse_labels = load_discovery_val_labels(
            corpus_dir=config.corpus_dir,
            remote_repo_id=config.remote_repo_id,
            remote_corpus_subpath=config.remote_corpus_subpath,
        )

    # Optimizer/scheduler are built AFTER the (possibly resumed) model so their param
    # references point at the loaded weights, then their state dicts are restored on top.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)

    scheduler = None
    if config.sae_type == "jumprelu":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        if scheduler is not None and resume_state["scheduler_state"] is not None:
            scheduler.load_state_dict(resume_state["scheduler_state"])

    train_dataset, train_loader = build_partition_dataloader(
        "discovery_train",
        batch_size=config.batch_size,
        cache_dir=config.cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=config.remote_subpath,
        corpus_dir=config.corpus_dir,
        remote_corpus_subpath=config.remote_corpus_subpath,
        max_cached_shards=config.max_cached_shards,
        num_workers=config.num_workers,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
    )
    _, val_loader = build_partition_dataloader(
        "discovery_val",
        batch_size=config.batch_size,
        cache_dir=config.cache_dir,
        remote_repo_id=config.remote_repo_id,
        remote_subpath=config.remote_subpath,
        corpus_dir=config.corpus_dir,
        remote_corpus_subpath=config.remote_corpus_subpath,
        max_cached_shards=config.max_cached_shards,
        num_workers=0,
        shuffle=False,
    )

    # Dead-latent census: token-windowed, owned by the training loop (not the model) so the
    # model itself stays a stateless, checkpoint-friendly transformers.PreTrainedModel.
    if resume_state is not None:
        tokens_since_fired = resume_state["tokens_since_fired"].to(device)
        step = resume_state["step"]
        epoch = resume_state["epoch"]
        best_val_mse = resume_state["best_val_mse"]
    else:
        tokens_since_fired = torch.zeros(config.d_hidden, device=device)
        step = 0
        epoch = 0
        best_val_mse = float("inf")

    pm.print(f"\n[SAE Training] sae_type={config.sae_type} device={device} dtype={config.dtype}")
    pm.print(f"  total_steps={config.total_steps} batch_size={config.batch_size}")
    if resume_state is not None:
        pm.print(f"  Resumed from {config.resume_from} at step {step} (epoch {epoch})")
    # Previous eval's alive-latent set (fired >=1x on discovery_val) — None until the first
    # eval completes. Its Jaccard overlap with the current eval's set tracks whether the
    # feature basis has stopped reorganizing (expected late) or is still shifting (expected
    # early).
    prev_alive_mask: torch.Tensor | None = None

    while step < config.total_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if step >= config.total_steps:
                break
            batch = batch.to(device)
            dead_latent_mask = tokens_since_fired > config.dead_latent_window_tokens

            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                out = _forward(config, model, batch, dead_latent_mask)
                loss = _total_loss(config, model, out, step)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            model.remove_decoder_gradient_parallel_component_()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # Captured right before the unit-norm constraint is reapplied: since every prior
            # step ended at norm 1, this mean is exactly how far this step's update pushed
            # decoder directions away from unit norm — a cheap per-step stability signal.
            decoder_pre_norm_mean = model.W_dec.norm(dim=-1).mean().item()
            model.normalize_decoder_()

            fired = out.latents.detach().gt(0).any(dim=0)
            tokens_since_fired += batch.shape[0]
            tokens_since_fired[fired] = 0.0

            step += 1

            if step % max(1, config.eval_interval_steps // 10) == 0:
                dead_frac = (
                    (tokens_since_fired > config.dead_latent_window_tokens).float().mean().item()
                )
                pm.render_card(
                    f"[SAE Training] {config.sae_type}",
                    [
                        f"step {step}/{config.total_steps}",
                        f"loss={loss.item():.4f}  mse={out.mse_loss.item():.4f}  l0={out.l0.item():.1f}",
                        f"dead={dead_frac:.1%}  decoder_pre_norm={decoder_pre_norm_mean:.3f}",
                        pm.get_vram_telemetry(),
                    ],
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/mse": out.mse_loss.item(),
                            "train/l0": out.l0.item(),
                            "train/dead_latent_fraction": dead_frac,
                            "train/decoder_pre_norm_mean": decoder_pre_norm_mean,
                        },
                        step=step,
                    )

            if step % config.eval_interval_steps == 0 or step == config.total_steps:
                eval_stats, alive_mask = _evaluate(model, val_loader, config, device)
                dead_frac = (
                    (tokens_since_fired > config.dead_latent_window_tokens).float().mean().item()
                )
                # Overlap with the previous eval's alive-latent set: 1.0 once the feature basis
                # has stopped reorganizing, expected to be lower earlier in training.
                if prev_alive_mask is not None:
                    union = (alive_mask | prev_alive_mask).sum().item()
                    alive_jaccard = (
                        (alive_mask & prev_alive_mask).sum().item() / union if union > 0 else 1.0
                    )
                else:
                    alive_jaccard = None
                prev_alive_mask = alive_mask

                pm.render_card(
                    f"[SAE Training] {config.sae_type} — eval @ step {step}",
                    [
                        f"val_mse={eval_stats['mse']:.4f}  explained_variance={eval_stats['explained_variance']:.1%}",
                        f"mean_l0={eval_stats['mean_l0']:.1f}  dead_latent_fraction={dead_frac:.1%}",
                        f"cosine_sim: mean={eval_stats['cosine_sim_mean']:.3f}  p10={eval_stats['cosine_sim_p10']:.3f}",
                        f"alive_latent_jaccard={'n/a' if alive_jaccard is None else f'{alive_jaccard:.1%}'}",
                    ],
                )
                if wandb_run is not None:
                    # feature_density_histogram is logged as a plain bin-count list (bins are
                    # fixed log10-spaced buckets over (1e-12, 1], set by density_histogram_bins)
                    # rather than a wandb.Histogram, to keep this path dependency-light.
                    wandb_run.log(
                        {f"val/{k}": v for k, v in eval_stats.items()}
                        | {"val/dead_latent_fraction": dead_frac}
                        | ({"val/alive_latent_jaccard": alive_jaccard} if alive_jaccard is not None else {}),
                        step=step,
                    )

                model.save_pretrained(checkpoint_dir / "latest")
                _save_resume_state(
                    checkpoint_dir / "latest",
                    step,
                    epoch,
                    optimizer,
                    scheduler,
                    tokens_since_fired,
                    best_val_mse,
                    wandb_run.id if wandb_run is not None else None,
                )
                if wandb_run is not None:
                    # Logged every eval interval (not just at the very end) so the cloud copy
                    # is always resume-ready — precisely when a Kaggle session might die.
                    _log_checkpoint_artifact(wandb_run, checkpoint_dir / "latest", config.sae_type, "latest")
                if config.save_all_checkpoints:
                    model.save_pretrained(checkpoint_dir / f"step_{step}")
                if eval_stats["mse"] < best_val_mse:
                    best_val_mse = eval_stats["mse"]
                    model.save_pretrained(checkpoint_dir / "best")
                    if wandb_run is not None:
                        _log_checkpoint_artifact(wandb_run, checkpoint_dir / "best", config.sae_type, "best")
                    pm.print(f"  New best checkpoint saved (val_mse={best_val_mse:.4f})")

            if collapse_labels is not None and (
                step % config.collapse_check_interval_steps == 0 or step == config.total_steps
            ):
                collapse_stats = run_collapse_check(model, collapse_labels, config, device)
                pm.print(f"\n  [collapse-check @ step {step}] (discovery_val only)")
                for stratum, stratum_stats in collapse_stats.items():
                    pm.print(
                        f"    {stratum}: mean_concentration_ratio="
                        f"{stratum_stats['mean_concentration_ratio']:.2f}  "
                        f"n_latents_above_2x_baseline={stratum_stats['n_latents_above_2x_baseline']:.0f}"
                        f" / n_active_latents={stratum_stats['n_active_latents']:.0f}"
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"collapse_check/{stratum}/{metric}": value
                            for stratum, stratum_stats in collapse_stats.items()
                            for metric, value in stratum_stats.items()
                        },
                        step=step,
                    )

        epoch += 1

    if wandb_run is not None:
        wandb_run.finish()

    pm.print(f"\n[SAE Training] Finished at step {step}. Checkpoints in {checkpoint_dir}")
    return {"final_step": step, "best_val_mse": best_val_mse, "checkpoint_dir": str(checkpoint_dir)}


def main():
    parser = argparse.ArgumentParser(
        description="Train a baseline SAE (TopK, JumpReLU, BatchTopK, or Gated)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous run's checkpoint_dir root, overriding config.resume_from — "
        "keeps one YAML config reusable for both a fresh and a resumed invocation.",
    )
    args = parser.parse_args()

    cfg = SAETrainingConfig.from_yaml(args.config)
    if args.resume_from is not None:
        cfg = cfg.model_copy(update={"resume_from": args.resume_from})
    run_sae_training(cfg)


if __name__ == "__main__":
    main()
