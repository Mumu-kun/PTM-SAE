"""Configuration schema for baseline SAE training runs."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class WandbConfig(BaseModel):
    """Optional Weights & Biases logging, disabled unless explicitly turned on."""

    enabled: bool = False
    project: str = "ptm-sae"
    run_name: str | None = None


class SAETrainingConfig(BaseModel):
    """Unified configuration for a single baseline SAE training run (one architecture, one
    hyperparameter combination — e.g. TopK k=32 width=4096). One YAML file per run."""

    # Architecture
    sae_type: Literal["topk", "jumprelu", "batchtopk", "gated"] = "topk"
    d_in: int = 1280
    d_hidden: int = 4096
    k: int = 32  # TopK / BatchTopK only
    bandwidth: float = 1e-3  # JumpReLU only
    init_threshold: float = 0.001  # JumpReLU only
    l0_coefficient: float | None = None  # JumpReLU only — required if sae_type == "jumprelu"
    l0_warmup_steps: int = 10_000  # JumpReLU only (Rajamanoharan et al., 2024: 10k steps / 40M tokens)
    l1_coefficient: float | None = None  # Gated only — required if sae_type == "gated"

    # AuxK dead-latent mitigation (TopK / BatchTopK only). None means "use that architecture's
    # own published default" (off for TopK, on at 1/32 for BatchTopK) — see TopKSAEConfig /
    # BatchTopKSAEConfig. Set explicitly here to make it an ablation parameter for either.
    auxk_coefficient: float | None = None
    k_aux: int | None = None
    threshold_ema_decay: float = 0.99  # BatchTopK only

    # Optimization — no silent defaults for total_steps: corpus-size-dependent, must be set per run.
    total_steps: int
    learning_rate: float = 4e-4
    # JumpReLU only (Rajamanoharan et al., 2024). TopK trains at a constant LR (Gao et al., 2024);
    # BatchTopK/Gated default to constant too — no cited evidence yet that a schedule helps either.
    warmup_steps: int = 1000
    grad_clip_norm: float | None = None  # only needed at large scale (Gao et al., 2024)
    batch_size: int = 4096
    dtype: Literal["fp32", "bf16"] = "fp32"
    seed: int = 0

    # Dead-latent census (token-windowed, not step-windowed — robust to batch size changes)
    dead_latent_window_tokens: int = 500_000

    # Evaluation and checkpointing
    eval_interval_steps: int = 500
    checkpoint_dir: str = "checkpoints/run"
    save_all_checkpoints: bool = False

    # Lightweight interpretability diagnostics. Tier 2 (density histogram, cosine similarity,
    # alive-latent stability) is pure tensor math on the existing eval pass and always runs.
    # Tier 3 (the Residue Collapse canary) additionally joins discovery_val activations against
    # ptm_sites.jsonl and is opt-in since it's a real (if coarse) new computation, not free.
    density_histogram_bins: int = 20
    enable_collapse_check: bool = False
    # None means "5x the main eval cadence" — the PTM-label join is heavier than a plain
    # reconstruction pass, so it runs less often by default.
    collapse_check_interval_steps: int | None = None

    # Data source (mirrors ActivationPartitionDataset / SafeTensorsReader)
    cache_dir: str = "cache/activations/esm2_650m_l24"
    remote_repo_id: str | None = "mustafa-muhaimin/ptm-sae-dataset"
    remote_subpath: str | None = "activations/esm2_t33_650M_UR50D/layer_24"
    corpus_dir: str = "data/processed"
    remote_corpus_subpath: str = "corpus"
    # LRU eviction below the full shard count guarantees re-downloading, not less disk usage:
    # every training epoch touches the entire partition in a freshly shuffled order, so any
    # shard evicted mid-epoch will simply be re-hydrated the next time that epoch's shuffle
    # reaches it. Only set this if disk genuinely cannot hold the full partition and repeated
    # downloads are an accepted tradeoff — leave it None (unlimited local retention) otherwise.
    max_cached_shards: int | None = None
    num_workers: int = 0
    shuffle_buffer_size: int = 65536

    # Logging
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    def model_post_init(self, __context) -> None:
        if self.sae_type == "jumprelu" and self.l0_coefficient is None:
            raise ValueError(
                "l0_coefficient must be set explicitly for sae_type='jumprelu' — "
                "it is the reconstruction/sparsity tradeoff knob and has no safe default."
            )
        if self.sae_type == "gated" and self.l1_coefficient is None:
            raise ValueError(
                "l1_coefficient must be set explicitly for sae_type='gated' — "
                "it is the gate's L1 penalty weight and has no safe default."
            )
        if self.collapse_check_interval_steps is None:
            self.collapse_check_interval_steps = 5 * self.eval_interval_steps

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SAETrainingConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
