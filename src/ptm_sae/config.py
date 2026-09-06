"""Configuration schema for PTM SAE Engine."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model_name: str = Field(default="facebook/esm2_t6_8M_UR50D")
    target_layer: int = Field(default=4)
    device: str = Field(default="auto")
    dtype: Literal["fp16", "bf16", "fp32"] = Field(default="fp16")

class ExtractionConfig(BaseModel):
    max_sequence_length: int = Field(default=1022)
    batch_size: int = Field(default=2)

class ShardingConfig(BaseModel):
    output_dir: str = Field(default="cache/activations/dev_8m")
    max_shard_bytes: int = Field(default=10485760)  # 10 MB default for dev
    resume: bool = Field(default=True)

class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
