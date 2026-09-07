"""Zero-copy memory-mapped SafeTensors reader with on-demand remote shard hydration and LRU eviction."""

import contextlib
import gc
import json
import time
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from safetensors import safe_open

from ptm_sae.extraction.hub import HfSyncClient


class SafeTensorsReader:
    """Provides fast zero-copy lookups of protein and residue activations from cached shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        remote_repo_id: str | None = None,
        remote_subpath: str | None = None,
        max_cached_shards: int | None = None,
        token: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"

        self.remote_repo_id = remote_repo_id
        self.remote_subpath = remote_subpath
        self.max_cached_shards = max_cached_shards
        self.hub: HfSyncClient | None = (
            HfSyncClient(token=token) if remote_repo_id else None
        )

        # 1. Fetch remote manifest if not cached locally
        if not self.manifest_path.exists() and self.hub and self.remote_repo_id:
            sub = self.remote_subpath.rstrip("/") if self.remote_subpath else None
            downloaded = self.hub.fetch_manifest(
                self.remote_repo_id, sub, self.manifest_path
            )
            if not downloaded:
                raise FileNotFoundError(
                    f"Manifest not found locally at {self.manifest_path} or on remote repository {self.remote_repo_id}."
                )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        # 2. Parse index manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.entries: dict[str, dict] = self.manifest.get("entries", {})
        self._open_shards: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}
        self._mean_pooled_tensors: dict[str, torch.Tensor] | None = None

    def _evict_lru(self, keep_shards: set[str]):
        """Evict oldest cached SafeTensors shards to keep local disk consumption within max_cached_shards."""
        if not self.max_cached_shards:
            return

        local_shards = [f.name for f in self.cache_dir.glob("shard_*.safetensors")]
        if len(local_shards) <= self.max_cached_shards:
            return

        # Sort candidate shards from oldest accessed to newest accessed
        candidates = sorted(
            [s for s in local_shards if s not in keep_shards],
            key=lambda s: self._access_times.get(s, 0.0),
        )

        excess = len(local_shards) - self.max_cached_shards
        for shard_name in candidates[:excess]:
            if shard_name in self._open_shards:
                del self._open_shards[shard_name]
            gc.collect()
            with contextlib.suppress(PermissionError, OSError):
                (self.cache_dir / shard_name).unlink(missing_ok=True)

    def _get_shard_handle(self, shard_filename: str):
        shard_path = self.cache_dir / shard_filename

        # 1. Hydrate shard on-demand from remote repository structured shards/ subpath
        if not shard_path.exists():
            if not (self.hub and self.remote_repo_id):
                raise FileNotFoundError(
                    f"Shard file {shard_filename} not found in {self.cache_dir}"
                )

            sub = self.remote_subpath.rstrip("/") if self.remote_subpath else ""
            shards_subpath = f"{sub}/shards" if sub else "shards"
            self.hub.hydrate_shard(
                self.remote_repo_id, shards_subpath, shard_filename, shard_path
            )
            self._evict_lru(keep_shards={shard_filename})

        # 2. Touch access timestamp for LRU accounting
        self._access_times[shard_filename] = time.time()

        # 3. Cache memory-mapped file handle
        if shard_filename not in self._open_shards:
            self._open_shards[shard_filename] = safe_open(
                str(shard_path), framework="pt", device="cpu"
            )

        return self._open_shards[shard_filename]

    @property
    def total_tokens(self) -> int:
        return self.manifest.get("total_tokens", 0)

    def list_proteins(self) -> list[str]:
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

        # Slice directly from zero-copy memory map and clone to release mmap handle
        activations_slice = shard.get_slice("activations")
        return activations_slice[start:end].clone()

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
        return activations_slice[offset].clone()

    def get_mean_pooled_embedding(self, uniprot_id: str) -> torch.Tensor:
        """Retrieve the global mean-pooled residue embedding for a protein."""
        mean_file = self.cache_dir / "mean_pooled_embeddings.safetensors"

        # Hydrate auxiliary embeddings from remote if missing locally
        if not mean_file.exists() and self.hub and self.remote_repo_id:
            sub = self.remote_subpath.rstrip("/") if self.remote_subpath else None
            with contextlib.suppress(Exception):
                self.hub.hydrate_shard(
                    self.remote_repo_id,
                    sub,
                    "mean_pooled_embeddings.safetensors",
                    mean_file,
                )

        # Lazy-load cached dictionary on first request
        if self._mean_pooled_tensors is None:
            if mean_file.exists():
                with contextlib.suppress(Exception):
                    self._mean_pooled_tensors = safetensors.torch.load_file(
                        str(mean_file)
                    )
                if self._mean_pooled_tensors is None:
                    self._mean_pooled_tensors = {}
            else:
                self._mean_pooled_tensors = {}

        if uniprot_id in self._mean_pooled_tensors:
            return self._mean_pooled_tensors[uniprot_id].clone()

        # Resilient fallback: compute dynamically from shard residue activations
        if uniprot_id in self.entries:
            acts = self.get_protein_activations(uniprot_id)
            mean_vec = acts.mean(dim=0).contiguous()
            self._mean_pooled_tensors[uniprot_id] = mean_vec
            return mean_vec.clone()

        raise KeyError(
            f"UniProt ID '{uniprot_id}' not found in manifest or mean-pooled cache."
        )

    def close(self):
        """Release open memory-map shard handles and cached tensors (Windows safety)."""
        self._open_shards.clear()
        self._mean_pooled_tensors = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
