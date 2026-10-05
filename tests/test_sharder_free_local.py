"""A shard is deleted from local disk once it is on the Hub, when the run asks for that."""

import pytest
import torch

from ptm_sae.config import ShardingConfig
from ptm_sae.extraction import sharder as sharder_module
from ptm_sae.extraction.sharder import SafeTensorsSharder


class FakeHub:
    def __init__(self, fail_uploads: bool = False):
        self.fail_uploads = fail_uploads
        self.uploaded: list[str] = []

    def fetch_manifest(self, *args, **kwargs):
        return False

    def hydrate_shard(self, *args, **kwargs):
        raise FileNotFoundError

    def upload_shard(self, repo_id, subpath, path):
        if self.fail_uploads:
            raise ConnectionError("hub unreachable")
        self.uploaded.append(path.name)
        return True

    def upload_manifest(self, *args, **kwargs):
        return True


def _write_one_shard(tmp_path, monkeypatch, hub, free_local):
    monkeypatch.setattr(sharder_module, "HfSyncClient", lambda: hub)
    config = ShardingConfig(
        output_dir=str(tmp_path), remote_repo_id="mock/acts", remote_subpath="a", free_local_after_upload=free_local
    )
    sharder = SafeTensorsSharder(config)
    sharder.add_protein("P1", torch.ones(3, 4), torch.ones(4))
    sharder.flush()
    for future in sharder._pending_uploads:
        future.result()
    sharder.close()
    return tmp_path / "shard_0000.safetensors"


def test_uploaded_shard_is_freed_when_asked(tmp_path, monkeypatch):
    hub = FakeHub()

    shard = _write_one_shard(tmp_path, monkeypatch, hub, free_local=True)

    assert hub.uploaded == ["shard_0000.safetensors", "mean_pooled_embeddings.safetensors"]
    assert not shard.exists()
    assert (tmp_path / "manifest.json").exists()  # the manifest stays: it is how a resume finds its place


def test_shard_is_kept_by_default(tmp_path, monkeypatch):
    assert _write_one_shard(tmp_path, monkeypatch, FakeHub(), free_local=False).exists()


def test_shard_is_kept_when_its_upload_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sharder_module, "HfSyncClient", lambda: FakeHub(fail_uploads=True))
    config = ShardingConfig(
        output_dir=str(tmp_path), remote_repo_id="mock/acts", remote_subpath="a", free_local_after_upload=True
    )
    sharder = SafeTensorsSharder(config)
    sharder.add_protein("P1", torch.ones(3, 4), torch.ones(4))
    sharder.flush()

    with pytest.raises(ConnectionError):
        sharder.close()

    assert (tmp_path / "shard_0000.safetensors").exists()  # still there for the next attempt
