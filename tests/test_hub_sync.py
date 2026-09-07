"""Unit tests for Hugging Face Hub synchronization, token cascade, and resilient shard hydration."""

import json
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch

from ptm_sae.config import ShardingConfig
from ptm_sae.extraction.hub import (
    HfSyncClient,
    resolve_hf_token,
    retry_with_backoff,
)
from ptm_sae.extraction.reader import SafeTensorsReader
from ptm_sae.extraction.sharder import SafeTensorsSharder


def test_resolve_hf_token_cascade(monkeypatch):
    # 1. Explicit token takes highest priority
    monkeypatch.setenv("HF_TOKEN", "env_token_val")
    assert resolve_hf_token("explicit_token_val") == "explicit_token_val"

    # 2. Falls back to HF_TOKEN env var
    assert resolve_hf_token(None) == "env_token_val"

    # 3. If env var is missing, falls back to get_token
    monkeypatch.delenv("HF_TOKEN")
    with patch("ptm_sae.extraction.hub.get_token", return_value="cached_token_val"):
        assert resolve_hf_token(None) == "cached_token_val"


def test_retry_with_backoff_success_and_failure():
    # Transient failures succeed on retry
    attempts = [0]

    def flakey_network():
        attempts[0] += 1
        if attempts[0] < 2:
            raise ConnectionError("Connection reset by peer")
        return "success"

    result = retry_with_backoff(
        flakey_network, max_retries=3, base_delay=0.01, backoff_factor=1.0
    )
    assert result == "success"
    assert attempts[0] == 2

    # Persistent failures raise RuntimeError
    def always_fail():
        raise TimeoutError("Socket timeout")

    with pytest.raises(RuntimeError, match="Operation failed after 2 attempts"):
        retry_with_backoff(
            always_fail, max_retries=2, base_delay=0.01, backoff_factor=1.0
        )


def test_hf_sync_client_path_normalization():
    assert (
        HfSyncClient.build_repo_path("esm2_650m/layer_24", "manifest.json")
        == "esm2_650m/layer_24/manifest.json"
    )
    assert (
        HfSyncClient.build_repo_path("/esm2_650m/layer_24/", "shard_0000.safetensors")
        == "esm2_650m/layer_24/shard_0000.safetensors"
    )
    assert HfSyncClient.build_repo_path(None, "manifest.json") == "manifest.json"


def test_hf_sync_hydrate_atomic_write_and_size_guard(tmp_path):
    client = HfSyncClient(token="mock_token")
    mock_source = tmp_path / "mock_downloaded.safetensors"
    safetensors.torch.save_file({"acts": torch.zeros(10, 32)}, mock_source)
    real_size = mock_source.stat().st_size

    target = tmp_path / "cache" / "shard_0000.safetensors"

    with patch("ptm_sae.extraction.hub.hf_hub_download", return_value=str(mock_source)):
        # Successful download and validation
        hydrated = client.hydrate_shard(
            repo_id="mock/repo",
            subpath="layer_4",
            shard_name="shard_0000.safetensors",
            target_path=target,
            expected_bytes=real_size,
        )
        assert hydrated.exists()
        assert hydrated.stat().st_size == real_size

        # Mismatched byte size raises error and removes corrupt target
        with pytest.raises(IOError, match="Corrupt shard download"):
            client.hydrate_shard(
                repo_id="mock/repo",
                subpath="layer_4",
                shard_name="shard_0001.safetensors",
                target_path=tmp_path / "cache" / "shard_0001.safetensors",
                expected_bytes=real_size + 999,
            )


def test_sharder_remote_upload_on_flush(tmp_path):
    cfg = ShardingConfig(
        output_dir=str(tmp_path / "shards"),
        max_shard_bytes=500_000,
        remote_repo_id="mock_user/ptm-activations",
        remote_subpath="esm2_8m/layer_4",
    )

    with patch("ptm_sae.extraction.sharder.HfSyncClient") as mock_hub_cls:
        mock_hub = MagicMock()
        mock_hub.fetch_manifest.return_value = False
        mock_hub_cls.return_value = mock_hub

        sharder = SafeTensorsSharder(cfg)

        # Add dummy protein and close (which flushes and drains background uploads)
        res_tensor = torch.zeros(50, 320)
        mean_vec = torch.zeros(320)
        sharder.add_protein("P12345", res_tensor, mean_vec)
        sharder.close()

        # Verify upload calls occurred
        assert mock_hub.upload_shard.called
        assert mock_hub.upload_manifest.called


def test_reader_remote_hydration_and_lru_eviction(tmp_path):
    cache_dir = tmp_path / "cache_lru"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy shards 0 and 1
    shard0_file = cache_dir / "shard_0000.safetensors"
    shard1_file = cache_dir / "shard_0001.safetensors"
    safetensors.torch.save_file({"activations": torch.ones(10, 8)}, shard0_file)
    safetensors.torch.save_file({"activations": torch.ones(10, 8) * 2}, shard1_file)

    manifest_data = {
        "version": "1.0",
        "total_tokens": 30,
        "shards": [
            "shard_0000.safetensors",
            "shard_0001.safetensors",
            "shard_0002.safetensors",
        ],
        "entries": {
            "P00001": {
                "length": 10,
                "shard_file": "shard_0000.safetensors",
                "start_offset": 0,
                "end_offset": 10,
            },
            "P00002": {
                "length": 10,
                "shard_file": "shard_0001.safetensors",
                "start_offset": 0,
                "end_offset": 10,
            },
            "P00003": {
                "length": 10,
                "shard_file": "shard_0002.safetensors",
                "start_offset": 0,
                "end_offset": 10,
            },
        },
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Initialize reader with max_cached_shards=2
    reader = SafeTensorsReader(
        cache_dir=cache_dir,
        remote_repo_id="mock_user/ptm-activations",
        remote_subpath="layer_4",
        max_cached_shards=2,
    )
    reader.hub = MagicMock(spec=HfSyncClient)

    # Mock hydration for shard_0002 when requested
    def mock_hydrate(repo_id, subpath, shard_name, target_path, expected_bytes=None):
        safetensors.torch.save_file({"activations": torch.ones(10, 8) * 3}, target_path)
        return target_path

    reader.hub.hydrate_shard.side_effect = mock_hydrate

    # Access P1 (shard 0) first, then P2 (shard 1)
    act1 = reader.get_protein_activations("P00001")
    assert act1.shape == (10, 8)
    act2 = reader.get_protein_activations("P00002")
    assert act2.shape == (10, 8)

    assert shard0_file.exists()
    assert shard1_file.exists()

    # Access P3 (shard 2), which downloads shard_2 and triggers LRU eviction since limit is 2
    # Shard 0 is the least-recently-used, so it should be evicted
    act3 = reader.get_protein_activations("P00003")
    assert act3.shape == (10, 8)

    shard2_file = cache_dir / "shard_0002.safetensors"
    assert shard2_file.exists()
    assert shard1_file.exists()
    assert not shard0_file.exists(), "Shard 0 was not evicted under LRU limit"
