"""Configuration schemas for PTM SAE Engine."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """PLM encoder configuration."""
    model_name: str = Field(default="facebook/esm2_t6_8M_UR50D", description="HuggingFace model identifier")
    target_layer: int = Field(default=4, description="Zero-indexed transformer layer to tap")
    device: str = Field(default="auto", description="Execution device ('cuda', 'cpu', or 'auto')")
    dtype: Literal["fp16", "bf16", "fp32"] = Field(default="fp16", description="Activation storage precision")


class ExtractionConfig(BaseModel):
    """Inference and token extraction batch parameters."""
    max_sequence_length: int = Field(default=1022, description="Maximum amino acid sequence length")
    batch_size: int = Field(default=2, description="Inference batch size")


class ShardingConfig(BaseModel):
    """SafeTensors shard buffer and manifest commit parameters."""
    output_dir: str = Field(default="cache/activations/dev_8m", description="Activation cache directory")
    max_shard_bytes: int = Field(default=10485760, description="Max raw bytes per SafeTensors shard before rotation")
    resume: bool = Field(default=True, description="Resume extraction from existing manifest commits")


class PipelineConfig(BaseModel):
    """Unified configuration container for full extraction and caching pipeline."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
