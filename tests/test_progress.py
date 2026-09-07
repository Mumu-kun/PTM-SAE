"""Unit tests for PipelineProgressManager in interactive and headless environments."""

import threading
from unittest.mock import MagicMock, patch

import torch

from ptm_sae.config import PipelineConfig, ShardingConfig
from ptm_sae.extraction.pipeline import run_extraction_pipeline
from ptm_sae.extraction.progress import PipelineProgressManager, PreformattedCard
from ptm_sae.extraction.sharder import SafeTensorsSharder
from ptm_sae.pipeline import run_full_lifecycle


def test_preformatted_card_rendering():
    card = PreformattedCard("┌─ PTM-SAE ─┐\n│ line 1    │\n└───────────┘")
    assert "<pre style=" in card._repr_html_()
    assert "┌─ PTM-SAE ─┐" in card._repr_html_()
    assert "line 1" in card
    assert str(card) == "┌─ PTM-SAE ─┐\n│ line 1    │\n└───────────┘"
    assert repr(card) == "┌─ PTM-SAE ─┐\n│ line 1    │\n└───────────┘"
    assert card == "┌─ PTM-SAE ─┐\n│ line 1    │\n└───────────┘"


def test_progress_manager_interactive_rendering():
    mock_display = MagicMock()
    mock_handle = MagicMock()
    mock_display.return_value = mock_handle

    manager = PipelineProgressManager(
        total_stages=5,
        is_interactive=True,
        display_fn=mock_display,
    )

    manager.start_pipeline()
    manager.start_stage(1, total_items=100, info="Acquisition")
    manager.update_stage(50, total=100)

    # Initial display call should have been made with display_id=True
    mock_display.assert_called_once()
    first_call_args = mock_display.call_args[0][0]
    assert "PTM-SAE Extraction Engine" in first_call_args
    assert "1. Data Acquisition" in first_call_args

    # Subsequent updates should use the handle update method to avoid terminal spam
    assert mock_handle.update.called
    update_call_args = mock_handle.update.call_args[0][0]
    assert "50%" in update_call_args

    manager.finish_stage(1, summary="Fetched 100 proteins")
    assert manager.stage_status[1] == "completed"


def test_progress_manager_headless_throttling():
    printed_lines = []

    def mock_print(msg, *args, **kwargs):
        printed_lines.append(str(msg))

    manager = PipelineProgressManager(
        total_stages=5,
        is_interactive=False,
        log_interval_pct=10.0,
    )
    manager.print = mock_print

    manager.start_stage(3, total_items=100, info="ESM-2 Extraction")

    # Simulate 100 fine-grained single item steps
    for i in range(1, 101):
        manager.update_stage(i, total=100)

    # In headless mode, updates should trigger only at 0%, 10%, 20% ... 100%
    update_lines = [line for line in printed_lines if "Progress:" in line]
    assert len(update_lines) <= 12, (
        f"Expected ~11 progress lines, got {len(update_lines)}"
    )
    assert any("10%" in line for line in update_lines)
    assert any("50%" in line for line in update_lines)
    assert any("100%" in line for line in update_lines)

    manager.finish_stage(3, summary="Extracted 100 proteins")
    assert any("Stage 3: ESM-2 Extraction" in line for line in printed_lines)


def test_vram_telemetry_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        manager = PipelineProgressManager()
        assert manager.get_vram_telemetry() == "VRAM: N/A"


def test_vram_telemetry_multi_gpu():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch(
            "torch.cuda.memory_allocated",
            side_effect=[3.2 * (1024**3), 3.4 * (1024**3)],
        ),
        patch("torch.cuda.get_device_properties") as mock_props,
        patch("torch.cuda.get_device_name", return_value="Tesla T4"),
    ):
        mock_prop = MagicMock()
        mock_prop.total_memory = 15.0 * (1024**3)
        mock_props.return_value = mock_prop

        manager = PipelineProgressManager()
        vram = manager.get_vram_telemetry()
        assert "GPU0: 3.2G" in vram
        assert "GPU1: 3.4G" in vram
        assert "15.0 GB" in vram


def test_stage_receipt_generation():
    printed_lines = []
    manager = PipelineProgressManager(is_interactive=False)
    manager.print = lambda msg, *args, **kwargs: printed_lines.append(str(msg))

    manager.start_stage(2, total_items=50, info="Clustering")
    manager.finish_stage(2, summary="Clustered 50 clusters")

    output = "\n".join(printed_lines)
    assert "Stage 2: Homology Split" in output
    assert "Summary: Clustered 50 clusters" in output
    assert "┌─ [✓]" in output


def test_thread_safe_concurrent_updates():
    manager = PipelineProgressManager(is_interactive=False)
    manager.print = lambda *args, **kwargs: None
    manager.start_stage(3, total_items=1000)

    def worker():
        for _ in range(100):
            manager.advance(1)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert manager.stage_current == 1000


def test_sharder_on_shard_flush_and_statistics(tmp_path):
    flushed_records = []

    def on_flush(idx: int, name: str):
        flushed_records.append((idx, name))

    cfg = ShardingConfig(
        output_dir=str(tmp_path / "shards"),
        max_shard_bytes=500,  # Small threshold to force flush
    )
    sharder = SafeTensorsSharder(cfg, on_shard_flush=on_flush)

    assert sharder.shard_count == 0
    assert sharder.buffer_bytes == 0
    assert sharder.buffer_tokens == 0

    # Add dummy protein
    t1 = torch.zeros(100, 32)
    m1 = torch.zeros(32)
    sharder.add_protein("P00001", t1, m1)
    sharder.close()

    assert len(flushed_records) >= 1
    assert flushed_records[0][0] == 0
    assert flushed_records[0][1] == "shard_0000.safetensors"
    assert sharder.shard_count >= 1


def test_extraction_pipeline_progress_integration(tmp_path):
    cfg = PipelineConfig.from_yaml("configs/dev_8m.yaml")
    cfg.sharding.output_dir = str(tmp_path / "cache_pipeline")
    cfg.sharding.max_shard_bytes = 200_000

    mock_pm = MagicMock(spec=PipelineProgressManager)

    manifest = run_extraction_pipeline(
        config=cfg,
        fasta_path="data/sample.fasta",
        progress_manager=mock_pm,
    )

    assert manifest["total_tokens"] == 685
    assert mock_pm.start_stage.called
    assert mock_pm.advance.called
    assert mock_pm.finish_stage.called
    # Stage 3 must be passed to start_stage and finish_stage
    assert any(call_args[0][0] == 3 for call_args in mock_pm.start_stage.call_args_list)
    assert any(
        call_args[0][0] == 3 for call_args in mock_pm.finish_stage.call_args_list
    )


def test_full_lifecycle_progress_integration(tmp_path):
    cfg = PipelineConfig.from_yaml("configs/dev_8m.yaml")
    cfg.sharding.output_dir = str(tmp_path / "lifecycle_cache")

    stages_started = []
    stages_finished = []

    class SpyProgressManager(PipelineProgressManager):
        def start_stage(self, stage_idx, total_items=0, info=""):
            stages_started.append(stage_idx)
            super().start_stage(stage_idx, total_items, info)

        def finish_stage(self, stage_idx, summary=""):
            stages_finished.append(stage_idx)
            super().finish_stage(stage_idx, summary)

    pm = SpyProgressManager(is_interactive=False)
    pm.print = lambda *args, **kwargs: None

    results = run_full_lifecycle(
        config=cfg,
        sample_only=True,
        progress_manager=pm,
    )

    assert "sharding_manifest" in results
    assert stages_started == [1, 2, 3, 4, 5]
    assert stages_finished == [1, 2, 3, 4, 5]
