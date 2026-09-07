"""Unified multi-stage progress tracking, telemetry, and output virtualization for PTM-SAE."""

import html
import sys
import threading
import time
from collections.abc import Callable
from typing import ClassVar

import torch


def _detect_ipython() -> bool:
    """Detect if execution environment is an interactive notebook (Jupyter/Colab/Kaggle)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:  # noqa: BLE001
        return False


class PreformattedCard:
    """Rich display object that renders cleanly in Jupyter/Kaggle notebooks and consoles.

    Implements Jupyter's rich display protocol (_repr_html_ and _repr_pretty_)
    to guarantee preformatted HTML rendering without raw string escaping or quotation marks.
    """

    def __init__(self, text: str):
        self.text = text

    def _repr_html_(self) -> str:
        escaped = html.escape(self.text)
        return (
            "<pre style='"
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; "
            "font-size: 13px; line-height: 1.35; padding: 10px 14px; margin: 6px 0; "
            "background-color: rgba(128, 128, 128, 0.07); border: 1px solid rgba(128, 128, 128, 0.2); "
            "border-radius: 6px; overflow-x: auto; color: inherit; display: block;"
            f"'>{escaped}</pre>"
        )

    def _repr_pretty_(self, p, cycle) -> None:
        p.text(self.text)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        return self.text == str(other)


class PipelineProgressManager:
    """Coordinates clean in-place progress dashboards and headless log throttling."""

    DEFAULT_STAGES: ClassVar[dict[int, str]] = {
        1: "Data Acquisition",
        2: "Homology Split",
        3: "ESM-2 Extraction",
        4: "Remote Hub Sync",
        5: "Verification",
    }

    def __init__(
        self,
        total_stages: int = 5,
        stage_names: dict[int, str] | None = None,
        is_interactive: bool | None = None,
        log_interval_pct: float = 10.0,
        enabled: bool = True,
        display_fn: Callable[..., None] | None = None,
        clear_fn: Callable[..., None] | None = None,
    ):
        self.total_stages = total_stages
        self.stage_names = stage_names or self.DEFAULT_STAGES.copy()
        self.enabled = enabled
        self.is_interactive = (
            _detect_ipython() if is_interactive is None else is_interactive
        )
        self.log_interval_pct = log_interval_pct

        # Lifecycle state
        self.current_stage: int = 0
        self.stage_total: int = 0
        self.stage_current: int = 0
        self.stage_status: dict[int, str] = {
            i: "pending" for i in range(1, total_stages + 1)
        }
        self.stage_timings: dict[int, float] = {}
        self.start_time: float | None = None
        self.stage_start_time: float | None = None

        # Telemetry metrics
        self.throughput: float = 0.0
        self.active_shard: int | None = None
        self.extra_info: str = ""
        self.last_reported_pct: float = -1.0

        # Concurrency safety
        self._lock = threading.Lock()

        # Output / display handles
        self._display_handle = None
        self._display_fn = display_fn
        self._clear_fn = clear_fn
        self._init_display_system()

    def _init_display_system(self) -> None:
        if not self.enabled:
            return

        if self._display_fn is None and self.is_interactive:
            try:
                from IPython.display import display

                self._display_fn = display
            except Exception:  # noqa: BLE001
                self._display_fn = None

    def print(self, *args, **kwargs) -> None:
        """Safe print that writes to sys.stdout and immediately flushes."""
        if not self.enabled:
            return
        kwargs["file"] = sys.stdout
        kwargs["flush"] = True
        print(*args, **kwargs)

    def get_vram_telemetry(self) -> str:
        """Format per-GPU allocated VRAM in GB."""
        if not torch.cuda.is_available():
            return "VRAM: N/A"
        try:
            device_count = torch.cuda.device_count()
            per_gpu = []
            for i in range(device_count):
                alloc_gb = torch.cuda.memory_allocated(i) / (1024**3)
                per_gpu.append(f"GPU{i}: {alloc_gb:.1f}G")
            total_max = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if device_count == 1:
                return f"VRAM: {per_gpu[0]} / {total_max:.1f} GB"
            return f"VRAM: [{' | '.join(per_gpu)}] / {total_max:.1f} GB"
        except Exception:  # noqa: BLE001
            return "GPU"

    def start_pipeline(self) -> None:
        """Mark overall pipeline beginning and reset timer."""
        with self._lock:
            self.start_time = time.perf_counter()
            self.current_stage = 0
            if not self.is_interactive:
                self.print("┌─ PTM-SAE Pipeline Execution Initialized ─────────────┐")

    def start_stage(self, stage_idx: int, total_items: int = 0, info: str = "") -> None:
        """Start tracking a specific pipeline stage."""
        with self._lock:
            self.current_stage = stage_idx
            self.stage_total = total_items
            self.stage_current = 0
            self.last_reported_pct = -1.0
            self.stage_status[stage_idx] = "running"
            self.stage_start_time = time.perf_counter()
            self.extra_info = info

            if self.is_interactive:
                self._render_interactive_dashboard()
            else:
                self.print(
                    f"\n[Stage {stage_idx}] {self.stage_names.get(stage_idx, '')} started {f'({info})' if info else ''}"
                )
                if total_items > 0:
                    self.print(f"  Total items to process: {total_items}")

    def update_stage(
        self,
        current: int,
        total: int | None = None,
        shard_idx: int | None = None,
        info: str | None = None,
    ) -> None:
        """Update progress metrics for the current active stage thread-safely."""
        with self._lock:
            self.stage_current = current
            if total is not None:
                self.stage_total = total
            if shard_idx is not None:
                self.active_shard = shard_idx
            if info is not None:
                self.extra_info = info

            # Calculate throughput
            if self.stage_start_time and current > 0:
                elapsed = time.perf_counter() - self.stage_start_time
                if elapsed > 0:
                    self.throughput = current / elapsed

            if self.is_interactive:
                self._render_interactive_dashboard()
            else:
                self._render_headless_tick(current)

    def advance(self, step: int = 1, info: str | None = None) -> None:
        """Convenience helper to increment current stage items by step."""
        with self._lock:
            new_val = self.stage_current + step
        self.update_stage(new_val, info=info)

    def set_active_shard(self, shard_idx: int) -> None:
        """Update active shard count and trigger display refresh."""
        with self._lock:
            self.active_shard = shard_idx
            if self.is_interactive:
                self._render_interactive_dashboard()

    def _render_headless_tick(self, current: int) -> None:
        """Throttled non-interactive progress updates emitted at interval percentages."""
        if self.stage_total > 0:
            pct = (current / self.stage_total) * 100.0
            if (
                self.last_reported_pct < 0
                or (pct - self.last_reported_pct) >= self.log_interval_pct
                or current >= self.stage_total
            ):
                threshold = int(pct // self.log_interval_pct) * int(
                    self.log_interval_pct
                )
                if threshold > self.last_reported_pct:
                    self.last_reported_pct = float(threshold)
                    rate_str = (
                        f" | {self.throughput:.1f} prot/s"
                        if self.throughput > 0
                        else ""
                    )
                    shard_str = (
                        f" | Shard #{self.active_shard}"
                        if self.active_shard is not None
                        else ""
                    )
                    vram_str = f" | {self.get_vram_telemetry()}"
                    self.print(
                        f"  [{self.stage_names.get(self.current_stage, 'Stage')}] "
                        f"Progress: {threshold:3d}% ({current}/{self.stage_total})"
                        f"{rate_str}{shard_str}{vram_str}"
                    )

    def finish_stage(self, stage_idx: int, summary: str = "") -> None:
        """Finalize stage, print persistent completion receipt, and update status."""
        with self._lock:
            self.stage_status[stage_idx] = "completed"
            elapsed = time.perf_counter() - (
                self.stage_start_time or time.perf_counter()
            )
            self.stage_timings[stage_idx] = elapsed

            # In interactive mode, render final state of dashboard
            if self.is_interactive:
                self._render_interactive_dashboard()

            elapsed_str = (
                f"{elapsed:.1f}s"
                if elapsed < 60
                else f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            )

            # Print persistent diagnostic summary receipt
            name = self.stage_names.get(stage_idx, f"Stage {stage_idx}")
            receipt_lines = [
                f"\n┌─ [✓] Stage {stage_idx}: {name} ({elapsed_str}) "
                + "─" * max(0, 45 - len(name)),
                f"│ Summary: {summary or 'Completed successfully'}",
            ]
            if stage_idx == 3 and self.active_shard is not None:
                receipt_lines.append(
                    f"│ Shards: {self.active_shard + 1} generated | Peak {self.get_vram_telemetry()}"
                )
            receipt_lines.append("└" + "─" * 60)
            self.print("\n".join(receipt_lines))

    def _generate_dashboard_card(self) -> str:
        """Construct the ASCII/Unicode dashboard view."""
        avg_rate = self.throughput

        lines = [
            "┌─ PTM-SAE Extraction Engine ──────────────────────────────────────────┐"
        ]
        for idx in range(1, 6):
            name = self.stage_names.get(idx, f"Stage {idx}")
            status = self.stage_status.get(idx, "pending")

            if status == "completed":
                t = self.stage_timings.get(idx, 0.0)
                t_str = f"{t:.1f}s" if t < 60 else f"{int(t // 60)}m {int(t % 60)}s"
                lines.append(
                    f"│ [✓] {idx}. {name:<18} (done in {t_str:<7})                     │"
                )
            elif status == "running":
                icon = "[▶]"
                if self.stage_total > 0:
                    pct = int((self.stage_current / self.stage_total) * 100)
                    pct = max(0, min(100, pct))
                    bar_len = 18
                    filled = int(bar_len * (pct / 100.0))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    lines.append(
                        f"│ {icon} {idx}. {name:<18}: [{bar}] {pct:3d}% ({self.stage_current}/{self.stage_total})"
                    )
                else:
                    lines.append(
                        f"│ {icon} {idx}. {name:<18}: In progress ({self.extra_info or 'active'})..."
                    )

                # Telemetry sub-lines
                rate_str = (
                    f"↳ Rate: {avg_rate:.1f} prot/s"
                    if avg_rate > 0
                    else "↳ Initializing..."
                )
                shard_str = (
                    f"Shard: #{self.active_shard:04d}"
                    if self.active_shard is not None
                    else "Shard: init"
                )
                lines.append(
                    f"│     {rate_str:<23}│ {shard_str:<12}│ {self.get_vram_telemetry():<23}│"
                )
            else:
                lines.append(
                    f"│ [ ] {idx}. {name:<18} (waiting)                                │"
                )

        lines.append(
            "└──────────────────────────────────────────────────────────────────────┘"
        )
        return "\n".join(lines)

    def _render_interactive_dashboard(self) -> None:
        """Render single dashboard card using IPython display update handle."""
        if not self.enabled or not self._display_fn:
            return

        card_text = self._generate_dashboard_card()
        rich_card = PreformattedCard(card_text)

        if self._display_handle is None:
            try:
                self._display_handle = self._display_fn(rich_card, display_id=True)
            except (TypeError, Exception):  # noqa: BLE001
                self._display_fn(rich_card)
        else:
            try:
                self._display_handle.update(rich_card)
            except Exception:  # noqa: BLE001
                self._display_fn(rich_card)
