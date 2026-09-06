"""SafeTensors sharded buffer manager and atomic manifest tracker."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import safetensors.torch
import torch

from ptm_sae.config import ShardingConfig


class SafeTensorsSharder:
    """Buffers residue activations and writes sharded SafeTensors with atomic manifest commits."""

    def __init__(self, sharding_cfg: ShardingConfig):
        self.cfg = sharding_cfg
        self.output_dir = Path(sharding_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"

        self.manifest: Dict = {
            "version": "1.0",
            "total_tokens": 0,
            "shards": [],
            "entries": {},
        }
        self.committed_ids = set()

        if self.cfg.resume and self.manifest_path.exists():
            self._load_manifest()

        # Find next available shard index
        self.current_shard_idx = len(self.manifest["shards"])

        # Active shard buffer
        self._buffer_tensors: List[torch.Tensor] = []
        self._buffer_entries: List[Dict] = []
        self._buffer_bytes: int = 0
        self._buffer_tokens: int = 0

        # Optional auxiliary storage
        self._mean_pooled_vectors: Dict[str, torch.Tensor] = {}

    def _load_manifest(self):
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.committed_ids = set(self.manifest.get("entries", {}).keys())
        except Exception:
            # If corrupted or empty, re-initialize
            self.manifest = {
                "version": "1.0",
                "total_tokens": 0,
                "shards": [],
                "entries": {},
            }
            self.committed_ids = set()

    def is_committed(self, uniprot_id: str) -> bool:
        return uniprot_id in self.committed_ids

    def add_protein(
        self,
        uniprot_id: str,
        residue_tensor: torch.Tensor,
        mean_pooled_vector: torch.Tensor,
    ):
        """Add a protein activation tensor to the shard buffer."""
        if self.is_committed(uniprot_id):
            return

        seq_len = residue_tensor.shape[0]
        tensor_bytes = residue_tensor.nelement() * residue_tensor.element_size()

        # Check if adding this protein exceeds shard size limit
        if (
            self._buffer_tensors
            and (self._buffer_bytes + tensor_bytes) > self.cfg.max_shard_bytes
        ):
            self.flush()

        start_offset = self._buffer_tokens
        end_offset = start_offset + seq_len

        self._buffer_tensors.append(residue_tensor)
        self._buffer_entries.append(
            {
                "uniprot_id": uniprot_id,
                "length": seq_len,
                "start_offset": start_offset,
                "end_offset": end_offset,
            }
        )
        self._buffer_bytes += tensor_bytes
        self._buffer_tokens += seq_len

        self._mean_pooled_vectors[uniprot_id] = mean_pooled_vector

    def flush(self):
        """Commit the current buffer to a SafeTensors shard and update manifest."""
        if not self._buffer_tensors:
            return

        shard_filename = f"shard_{self.current_shard_idx:04d}.safetensors"
        shard_path = self.output_dir / shard_filename

        # Concatenate along token dimension -> (total_tokens_in_shard, hidden_dim)
        contiguous_shard = torch.cat(self._buffer_tensors, dim=0).contiguous()
        safetensors.torch.save_file({"activations": contiguous_shard}, shard_path)

        # Update manifest entries
        for entry in self._buffer_entries:
            u_id = entry["uniprot_id"]
            self.manifest["entries"][u_id] = {
                "uniprot_id": u_id,
                "length": entry["length"],
                "shard_file": shard_filename,
                "start_offset": entry["start_offset"],
                "end_offset": entry["end_offset"],
            }
            self.committed_ids.add(u_id)

        if shard_filename not in self.manifest["shards"]:
            self.manifest["shards"].append(shard_filename)

        self.manifest["total_tokens"] += self._buffer_tokens

        # Flush optional mean-pooled vectors if any
        if self._mean_pooled_vectors:
            mean_path = self.output_dir / "mean_pooled_embeddings.safetensors"
            mean_dict = {k: v.contiguous() for k, v in self._mean_pooled_vectors.items()}
            if mean_path.exists():
                existing = safetensors.torch.load_file(mean_path)
                existing.update(mean_dict)
                safetensors.torch.save_file(existing, mean_path)
            else:
                safetensors.torch.save_file(mean_dict, mean_path)
            self._mean_pooled_vectors.clear()

        # Atomic manifest write
        temp_manifest = self.manifest_path.with_suffix(".tmp")
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(temp_manifest, self.manifest_path)

        # Reset buffer for next shard
        self.current_shard_idx += 1
        self._buffer_tensors = []
        self._buffer_entries = []
        self._buffer_bytes = 0
        self._buffer_tokens = 0

    def close(self):
        """Ensure all remaining buffered activations are flushed."""
        self.flush()
