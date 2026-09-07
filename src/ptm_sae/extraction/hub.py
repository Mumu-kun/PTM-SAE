"""Hugging Face Hub synchronization, zero-knowledge token discovery, and resilient shard transfers."""

import hashlib
import logging
import os
import shutil
import sys
import time
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T")


class SuppressEmptyCommitFilter(logging.Filter):
    """Filter that suppresses benign 'No files have been modified since last commit' warnings.

    Prevents Kaggle's red stderr alert boxes while letting genuine warnings and errors through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        return "No files have been modified since last commit" not in msg


def configure_hub_logging() -> None:
    """Installs SuppressEmptyCommitFilter on huggingface_hub loggers to prevent red stderr noise."""
    hub_filter = SuppressEmptyCommitFilter()
    for name in ("huggingface_hub", "huggingface_hub.hf_api", "huggingface_hub.utils"):
        target_logger = logging.getLogger(name)
        if not any(
            isinstance(f, SuppressEmptyCommitFilter) for f in target_logger.filters
        ):
            target_logger.addFilter(hub_filter)


# Configure filter automatically on module import
configure_hub_logging()


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a local file in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_hf_token(token: str | None = None) -> str | None:
    """Discovers Hugging Face authentication token across a 5-tier zero-knowledge cascade:

    1. Explicit function argument.
    2. Environment variable (HF_TOKEN).
    3. Google Colab Secrets (userdata.get('HF_TOKEN')).
    4. Kaggle Secrets (UserSecretsClient().get_secret('HF_TOKEN')).
    5. Local OS-level cache token (~/.cache/huggingface/token).
    """
    # 1. Explicit token argument
    if token:
        return token.strip()

    # 2. Environment variable
    if env_tok := os.environ.get("HF_TOKEN"):
        return env_tok.strip()

    # 3. Google Colab Secrets
    if "google.colab" in sys.modules or Path("/content").exists():
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            if colab_tok := userdata.get("HF_TOKEN"):
                return colab_tok.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    # 4. Kaggle Secrets
    if "kaggle_secrets" in sys.modules or Path("/kaggle").exists():
        try:
            from kaggle_secrets import (
                UserSecretsClient,  # type: ignore[import-not-found]
            )

            if kaggle_tok := UserSecretsClient().get_secret("HF_TOKEN"):
                return kaggle_tok.strip()
        except Exception:  # noqa: BLE001, S110
            pass

    # 5. Local OS-level cached token
    return get_token()


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff_factor: float = 2.5,
) -> T:
    """Executes a callable with exponential backoff for transient network and rate-limit errors.

    Bypasses retry for non-recoverable client authentication or missing resource errors.
    """
    last_error: Exception | None = None
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except HfHubHTTPError as e:
            # 401 Unauthorized, 403 Forbidden, 404 Not Found are non-recoverable
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403, 404):
                raise
            last_error = e
        except (ConnectionError, TimeoutError, urllib.error.URLError) as e:
            last_error = e
        except Exception as e:
            # Check for transient network error substrings
            msg = str(e).lower()
            if any(
                k in msg
                for k in ("connection", "timeout", "timed out", "reset", "closed")
            ):
                last_error = e
            else:
                raise

        if attempt == max_retries:
            break

        time.sleep(delay)
        delay *= backoff_factor

    raise RuntimeError(
        f"Operation failed after {max_retries} attempts: {last_error}"
    ) from last_error


class HfSyncClient:
    """Coordinates resilient SafeTensors shard uploads, downloads, and manifest tracking."""

    def __init__(self, token: str | None = None):
        self.token = resolve_hf_token(token)
        self.api = HfApi(token=self.token)
        self._known_hashes: dict[tuple[str, str], str] = {}
        configure_hub_logging()

    @staticmethod
    def build_repo_path(subpath: str | None, filename: str) -> str:
        """Construct normalized remote POSIX path inside dataset repository."""
        if not subpath:
            return filename
        clean_sub = subpath.strip("/").replace("\\", "/")
        return f"{clean_sub}/{filename}"

    def get_remote_file_sha256(self, repo_id: str, remote_path: str) -> str | None:
        """Fetch the SHA-256 digest of a remote file if present and indexed via LFS."""
        key = (repo_id, remote_path)
        if key in self._known_hashes:
            return self._known_hashes[key]

        try:
            paths_info = self.api.get_paths_info(
                repo_id=repo_id,
                paths=[remote_path],
                repo_type="dataset",
                token=self.token,
            )
            if paths_info and paths_info[0].lfs:
                sha = paths_info[0].lfs.sha256
                self._known_hashes[key] = sha
                return sha
        except Exception:  # noqa: BLE001
            return None
        return None

    def is_remote_file_identical(
        self,
        repo_id: str,
        remote_path: str,
        local_path: Path,
    ) -> bool:
        """Check if local file matches known remote SHA-256 before initiating commit."""
        key = (repo_id, remote_path)
        local_sha = compute_sha256(local_path)

        # 1. Check in-memory hash cache
        if key in self._known_hashes and self._known_hashes[key] == local_sha:
            return True

        # 2. Query remote repository metadata
        remote_sha = self.get_remote_file_sha256(repo_id, remote_path)
        return bool(remote_sha and remote_sha == local_sha)

    def fetch_manifest(
        self,
        repo_id: str,
        subpath: str | None,
        target_path: Path,
    ) -> bool:
        """Download remote manifest.json to local destination.

        Returns True if downloaded successfully, False if file does not exist on remote or auth fails.
        """
        remote_path = self.build_repo_path(subpath, "manifest.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=self.token,
            )
            shutil.copy2(cached_path, target_path)
            return True
        except Exception:  # noqa: BLE001
            return False

    def upload_shard(
        self,
        repo_id: str,
        subpath: str | None,
        file_path: Path,
        check_hash: bool = True,
    ) -> str | None:
        """Upload a SafeTensors shard or auxiliary embedding file with exponential backoff.

        Pre-checks SHA-256 against remote state to avoid redundant upload calls when possible.
        """
        remote_path = self.build_repo_path(subpath, file_path.name)

        if check_hash and self.is_remote_file_identical(
            repo_id, remote_path, file_path
        ):
            return None

        def _upload():
            return self.api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
            )

        res = retry_with_backoff(_upload, max_retries=3, base_delay=2.0)
        self._known_hashes[(repo_id, remote_path)] = compute_sha256(file_path)
        return str(res)

    def upload_manifest(
        self,
        repo_id: str,
        subpath: str | None,
        manifest_path: Path,
        check_hash: bool = True,
    ) -> str | None:
        """Atomically commit manifest.json to the remote dataset repository."""
        remote_path = self.build_repo_path(subpath, "manifest.json")

        if check_hash and self.is_remote_file_identical(
            repo_id, remote_path, manifest_path
        ):
            return None

        def _upload():
            return self.api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
            )

        res = retry_with_backoff(_upload, max_retries=3, base_delay=2.0)
        self._known_hashes[(repo_id, remote_path)] = compute_sha256(manifest_path)
        return str(res)

    def hydrate_shard(
        self,
        repo_id: str,
        subpath: str | None,
        shard_name: str,
        target_path: Path,
        expected_bytes: int | None = None,
    ) -> Path:
        """Download a single SafeTensors shard on-demand via atomic temporary file renaming.

        Guarantees that partially downloaded or corrupted files are never retained.
        """
        remote_path = self.build_repo_path(subpath, shard_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = target_path.with_suffix(f".tmp_{os.getpid()}")

        def _download():
            return hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=self.token,
            )

        cached_file = retry_with_backoff(_download, max_retries=3, base_delay=2.0)

        # 1. Copy downloaded file to atomic temporary path
        shutil.copy2(cached_file, tmp_dest)

        # 2. Verify expected byte size if provided
        actual_bytes = tmp_dest.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            tmp_dest.unlink(missing_ok=True)
            raise OSError(
                f"Corrupt shard download: {shard_name} expected {expected_bytes} bytes, got {actual_bytes} bytes."
            )

        # 3. Atomically replace target destination
        os.replace(tmp_dest, target_path)
        return target_path
