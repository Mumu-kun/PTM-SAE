"""SafeTensors sharded buffer manager with asynchronous background upload and thread safety."""

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import safetensors.torch
import torch

from ptm_sae.config import ShardingConfig
from ptm_sae.extraction.hub import HfSyncClient


class SafeTensorsSharder:
    """Buffers residue activations and writes sharded SafeTensors with atomic manifest commits."""

    def __init__(self, sharding_cfg: ShardingConfig):
        self.cfg = sharding_cfg
        self.output_dir = Path(sharding_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"

        # Thread safety lock for multi-GPU worker ingestion
        self._lock = threading.Lock()

        # Initialize remote sync client and async upload pool if repository is configured
        self.hub: HfSyncClient | None = (
            HfSyncClient() if sharding_cfg.remote_repo_id else None
        )
        self._upload_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_upload")
            if sharding_cfg.remote_repo_id
            else None
        )
        self._pending_uploads: list[Future] = []

        self.manifest: dict = {
            "version": "1.0",
            "total_tokens": 0,
            "shards": [],
            "entries": {},
        }
        self.committed_ids = set()

        # Resumption: sync remote manifest if local manifest is missing
        if (
            self.cfg.resume
            and not self.manifest_path.exists()
            and self.hub
            and self.cfg.remote_repo_id
        ):
            sub = (
                self.cfg.remote_subpath.rstrip("/") if self.cfg.remote_subpath else None
            )
            self.hub.fetch_manifest(self.cfg.remote_repo_id, sub, self.manifest_path)

        if self.cfg.resume and self.manifest_path.exists():
            self._load_manifest()

        # Find next available shard index
        self.current_shard_idx = len(self.manifest["shards"])

        # Active shard buffer
        self._buffer_tensors: list[torch.Tensor] = []
        self._buffer_entries: list[dict] = []
        self._buffer_bytes: int = 0
        self._buffer_tokens: int = 0

        # Auxiliary mean-pooled sequence storage
        self._mean_pooled_vectors: dict[str, torch.Tensor] = {}

    def _load_manifest(self):
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.committed_ids = set(self.manifest.get("entries", {}).keys())
        except (json.JSONDecodeError, OSError, KeyError):
            # Re-initialize cleanly if file is empty or unparseable
            self.manifest = {
                "version": "1.0",
                "total_tokens": 0,
                "shards": [],
                "entries": {},
            }
            self.committed_ids = set()

    def is_committed(self, uniprot_id: str) -> bool:
        with self._lock:
            return uniprot_id in self.committed_ids

    def add_protein(
        self,
        uniprot_id: str,
        residue_tensor: torch.Tensor,
        mean_pooled_vector: torch.Tensor,
    ):
        """Add a protein activation tensor and its mean-pooled context to the shard buffer (thread-safe)."""
        with self._lock:
            if uniprot_id in self.committed_ids:
                return

            seq_len = residue_tensor.shape[0]
            tensor_bytes = residue_tensor.nelement() * residue_tensor.element_size()

            # Flush buffer before adding if adding would exceed max shard byte limit
            if (
                self._buffer_tensors
                and (self._buffer_bytes + tensor_bytes) > self.cfg.max_shard_bytes
            ):
                self._flush_internal()

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
        """Commit the current buffer to a SafeTensors shard and update manifest (thread-safe)."""
        with self._lock:
            self._flush_internal()

    def _flush_internal(self):
        if not self._buffer_tensors:
            return

        shard_filename = f"shard_{self.current_shard_idx:04d}.safetensors"
        shard_path = self.output_dir / shard_filename

        # 1. Write contiguous token shard along token dimension (total_tokens, hidden_dim)
        contiguous_shard = torch.cat(self._buffer_tensors, dim=0).contiguous()
        safetensors.torch.save_file({"activations": contiguous_shard}, shard_path)

        # 2. Register shard metadata and offsets in manifest
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

        # 3. Append auxiliary mean-pooled vectors to separate single file
        mean_path = self.output_dir / "mean_pooled_embeddings.safetensors"
        if self._mean_pooled_vectors:
            mean_dict = {
                k: v.contiguous() for k, v in self._mean_pooled_vectors.items()
            }
            existing = (
                safetensors.torch.load_file(mean_path) if mean_path.exists() else {}
            )
            existing.update(mean_dict)
            safetensors.torch.save_file(existing, mean_path)
            self._mean_pooled_vectors.clear()

        # 4. Atomically commit manifest and reset active buffer
        temp_manifest = self.manifest_path.with_suffix(".tmp")
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(temp_manifest, self.manifest_path)

        # 5. Asynchronous background upload to remote authority (non-blocking)
        if self.hub and self.cfg.remote_repo_id:
            hub_client = self.hub
            remote_repo = self.cfg.remote_repo_id
            sub = self.cfg.remote_subpath.rstrip("/") if self.cfg.remote_subpath else ""
            shards_subpath = f"{sub}/shards" if sub else "shards"

            def _async_upload(
                s_path=shard_path, m_path=mean_path, man_path=self.manifest_path
            ):
                hub_client.upload_shard(remote_repo, shards_subpath, s_path)
                if m_path.exists():
                    hub_client.upload_shard(remote_repo, sub or None, m_path)
                hub_client.upload_manifest(remote_repo, sub or None, man_path)

            if self._upload_pool:
                fut = self._upload_pool.submit(_async_upload)
                self._pending_uploads.append(fut)
            else:
                _async_upload()

        self.current_shard_idx += 1
        self._buffer_tensors = []
        self._buffer_entries = []
        self._buffer_bytes = 0
        self._buffer_tokens = 0

    def close(self):
        """Ensure all remaining buffered activations are flushed and background uploads finish."""
        self.flush()

        # Wait for all background upload tasks to complete
        for fut in self._pending_uploads:
            fut.result()

        if self._upload_pool:
            self._upload_pool.shutdown(wait=True)
            self._upload_pool = None
