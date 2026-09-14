"""Training loop for baseline TopK / JumpReLU SAEs.

Streams discovery_train via ActivationPartitionDataset, trains with AdamW under each
architecture's literature-matched recipe (TopK: constant LR, Gao et al. 2024; JumpReLU:
warmup+cosine LR plus a separately warmed-up L0 coefficient, Rajamanoharan et al. 2024),
maintains the unit-norm decoder constraint each step, tracks a token-windowed dead-latent
census, and periodically evaluates on discovery_val with local-only checkpointing
(`latest` + `best`, matching a run's own `checkpoint_dir`).
"""

import argparse
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from ptm_sae.extraction.progress import PipelineProgressManager
from ptm_sae.models import (
    JumpReLUSAEConfig,
    JumpReLUSAEModel,
    SAEOutput,
    TopKSAEConfig,
    TopKSAEModel,
)
from ptm_sae.training.config import SAETrainingConfig
from ptm_sae.training.dataset import build_partition_dataloader


def _build_model(config: SAETrainingConfig) -> TopKSAEModel | JumpReLUSAEModel:
    if config.sae_type == "topk":
        return TopKSAEModel(
            TopKSAEConfig(d_in=config.d_in, d_hidden=config.d_hidden, k=config.k)
        )
    return JumpReLUSAEModel(
        JumpReLUSAEConfig(
            d_in=config.d_in,
            d_hidden=config.d_hidden,
            bandwidth=config.bandwidth,
            init_threshold=config.init_threshold,
        )
    )


def _total_loss(config: SAETrainingConfig, out: SAEOutput, step: int) -> torch.Tensor:
    if config.sae_type == "topk":
        return out.mse_loss
    # L0 coefficient is linearly warmed up over its own schedule, separate from the LR warmup
    # (Rajamanoharan et al., 2024 warm it up over 10k steps / 40M tokens to avoid collapsing
    # every latent to zero before the SAE has learned anything reconstructive).
    ramped_l0_coef = config.l0_coefficient * min(1.0, step / max(1, config.l0_warmup_steps))
    return out.mse_loss + ramped_l0_coef * out.l0


class _StreamingEvalStats:
    """Single-pass, per-dimension running statistics for MSE / explained variance / L0."""

    def __init__(self, d_in: int, device: torch.device):
        self.sum_x = torch.zeros(d_in, device=device)
        self.sum_x2 = torch.zeros(d_in, device=device)
        self.sum_sq_err = torch.tensor(0.0, device=device)
        self.sum_l0 = torch.tensor(0.0, device=device)
        self.n_rows = 0

    def update(self, x: torch.Tensor, reconstruction: torch.Tensor, l0: torch.Tensor) -> None:
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += (x**2).sum(dim=0)
        self.sum_sq_err += (x - reconstruction).pow(2).sum()
        self.sum_l0 += l0 * x.shape[0]
        self.n_rows += x.shape[0]

    def finalize(self) -> dict[str, float]:
        n = max(1, self.n_rows)
        mean_x = self.sum_x / n
        var_x_per_dim = (self.sum_x2 / n) - mean_x**2
        ss_tot = (var_x_per_dim.clamp_min(0) * n).sum()
        mse = (self.sum_sq_err / n).item()
        explained_variance = 1.0 - (self.sum_sq_err / ss_tot.clamp_min(1e-8)).item()
        return {
            "mse": mse,
            "explained_variance": explained_variance,
            "mean_l0": (self.sum_l0 / n).item(),
        }


@torch.no_grad()
def _evaluate(
    model: TopKSAEModel | JumpReLUSAEModel,
    val_loader,
    config: SAETrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Runs one deterministic pass over discovery_val. `val_loader` is built once by the
    caller and reused across every eval checkpoint — discovery_val never needs reshuffling."""
    model.eval()
    stats = _StreamingEvalStats(config.d_in, device)
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch)
        stats.update(batch, out.reconstruction, out.l0)
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

    wandb_run = None
    if config.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            config=config.model_dump(),
        )

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)

    scheduler = None
    if config.sae_type == "jumprelu":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

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
    tokens_since_fired = torch.zeros(config.d_hidden, device=device)

    pm.print(f"\n[SAE Training] sae_type={config.sae_type} device={device} dtype={config.dtype}")
    pm.print(f"  total_steps={config.total_steps} batch_size={config.batch_size}")

    best_val_mse = float("inf")
    step = 0
    epoch = 0

    while step < config.total_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if step >= config.total_steps:
                break
            batch = batch.to(device)

            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                out = model(batch)
                loss = _total_loss(config, out, step)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            model.remove_decoder_gradient_parallel_component_()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.normalize_decoder_()

            fired = out.latents.detach().gt(0).any(dim=0)
            tokens_since_fired += batch.shape[0]
            tokens_since_fired[fired] = 0.0

            step += 1

            if step % max(1, config.eval_interval_steps // 10) == 0:
                dead_frac = (
                    (tokens_since_fired > config.dead_latent_window_tokens).float().mean().item()
                )
                pm.print(
                    f"  step {step}/{config.total_steps}  loss={loss.item():.4f}  "
                    f"mse={out.mse_loss.item():.4f}  l0={out.l0.item():.1f}  "
                    f"dead={dead_frac:.1%}  {pm.get_vram_telemetry()}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/mse": out.mse_loss.item(),
                            "train/l0": out.l0.item(),
                            "train/dead_latent_fraction": dead_frac,
                        },
                        step=step,
                    )

            if step % config.eval_interval_steps == 0 or step == config.total_steps:
                eval_stats = _evaluate(model, val_loader, config, device)
                dead_frac = (
                    (tokens_since_fired > config.dead_latent_window_tokens).float().mean().item()
                )
                pm.print(
                    f"\n  [eval @ step {step}] val_mse={eval_stats['mse']:.4f}  "
                    f"explained_variance={eval_stats['explained_variance']:.1%}  "
                    f"mean_l0={eval_stats['mean_l0']:.1f}  dead_latent_fraction={dead_frac:.1%}\n"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/{k}": v for k, v in eval_stats.items()}
                        | {"val/dead_latent_fraction": dead_frac},
                        step=step,
                    )

                model.save_pretrained(checkpoint_dir / "latest")
                if config.save_all_checkpoints:
                    model.save_pretrained(checkpoint_dir / f"step_{step}")
                if eval_stats["mse"] < best_val_mse:
                    best_val_mse = eval_stats["mse"]
                    model.save_pretrained(checkpoint_dir / "best")
                    pm.print(f"  New best checkpoint saved (val_mse={best_val_mse:.4f})")

        epoch += 1

    if wandb_run is not None:
        wandb_run.finish()

    pm.print(f"\n[SAE Training] Finished at step {step}. Checkpoints in {checkpoint_dir}")
    return {"final_step": step, "best_val_mse": best_val_mse, "checkpoint_dir": str(checkpoint_dir)}


def main():
    parser = argparse.ArgumentParser(description="Train a baseline SAE (TopK or JumpReLU)")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    args = parser.parse_args()

    cfg = SAETrainingConfig.from_yaml(args.config)
    run_sae_training(cfg)


if __name__ == "__main__":
    main()
