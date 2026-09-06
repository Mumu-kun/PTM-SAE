"""Zero-copy memory-mapped SafeTensors reader."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import safetensors.torch
import torch
from safetensors import safe_open


class SafeTensorsReader:
    """Provides fast zero-copy lookups of protein and residue activations from cached shards."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.entries: Dict[str, Dict] = self.manifest.get("entries", {})
        self._open_shards: Dict[str, any] = {}
        self._mean_pooled_tensors: Optional[Dict[str, torch.Tensor]] = None

    def _get_shard_handle(self, shard_filename: str):
        # Cache memory-mapped file handles to avoid repeated syscall overhead
        if shard_filename not in self._open_shards:
            shard_path = str(self.cache_dir / shard_filename)
            self._open_shards[shard_filename] = safe_open(shard_path, framework="pt", device="cpu")
        return self._open_shards[shard_filename]

    @property
    def total_tokens(self) -> int:
        return self.manifest.get("total_tokens", 0)

    def list_proteins(self) -> List[str]:
        return list(self.entries.keys())

    def get_protein_activations(self, uniprot_id: str) -> torch.Tensor:
        """
        Retrieve activations for an entire protein sequence.
        Returns tensor of shape (L, hidden_dim) where index i maps strictly to biological residue i + 1.
        """
        if uniprot_id not in self.entries:
            raise KeyError(f"UniProt ID '{uniprot_id}' not found in manifest.")

        entry = self.entries[uniprot_id]
        shard = self._get_shard_handle(entry["shard_file"])
        start = entry["start_offset"]
        end = entry["end_offset"]

        # Slice directly from zero-copy memory map
        activations_slice = shard.get_slice("activations")
        return activations_slice[start:end]

    def get_residue_activation(self, uniprot_id: str, position: int) -> torch.Tensor:
        """
        Retrieve activation vector for a specific biological residue coordinate (1-indexed).
        position=1 returns the first amino acid activation (dim: hidden_dim).
        """
        if uniprot_id not in self.entries:
            raise KeyError(f"UniProt ID '{uniprot_id}' not found in manifest.")

        entry = self.entries[uniprot_id]
        length = entry["length"]
        if position < 1 or position > length:
            raise IndexError(
                f"Residue position {position} out of bounds for protein {uniprot_id} (length {length})."
            )

        # 1-indexed position maps to 0-indexed offset (position - 1)
        offset = entry["start_offset"] + position - 1
        shard = self._get_shard_handle(entry["shard_file"])
        activations_slice = shard.get_slice("activations")
        return activations_slice[offset]

    def get_mean_pooled_embedding(self, uniprot_id: str) -> torch.Tensor:
        """Retrieve the global mean-pooled residue embedding for a protein."""
        mean_file = self.cache_dir / "mean_pooled_embeddings.safetensors"
        if not mean_file.exists():
            raise FileNotFoundError(f"mean_pooled_embeddings.safetensors not found at {mean_file}")

        # Lazy-load cached dictionary on first request
        if self._mean_pooled_tensors is None:
            self._mean_pooled_tensors = safetensors.torch.load_file(str(mean_file))

        if uniprot_id not in self._mean_pooled_tensors:
            raise KeyError(f"UniProt ID '{uniprot_id}' not found in mean-pooled cache.")

        return self._mean_pooled_tensors[uniprot_id]
