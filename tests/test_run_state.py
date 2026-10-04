"""Guarded notebook cells, logged subprocesses and preflight: nothing here may stop a run."""

import json
import sys

import pytest

from ptm_sae import runtime


def test_failing_cell_is_recorded_and_the_run_continues(tmp_path, capsys):
    state = runtime.RunState(tmp_path / "state.json")

    with state.cell("good"):
        pass
    with state.cell("bad"):
        raise ValueError("silly error")
    with state.cell("dependent"):
        state.require("good", "bad")
        pytest.fail("a cell whose prerequisite failed must be skipped, not run")
    with state.cell("independent"):
        state.require("good")

    assert {n: c["status"] for n, c in state.cells.items()} == {
        "good": "ok",
        "bad": "failed",
        "dependent": "skipped",
        "independent": "ok",
    }
    assert state.cells["bad"]["detail"] == "ValueError: silly error"
    assert "needs ['bad']" in state.cells["dependent"]["detail"]
    out = capsys.readouterr().out
    assert "[FAILED] bad" in out and "[skip] dependent" in out
    # The history survives on disk, so a killed run still leaves it behind.
    on_disk = json.loads((tmp_path / "state.json").read_text())["cells"]
    assert on_disk["bad"]["status"] == "failed"


def test_interrupts_are_recorded_but_still_stop_the_run(tmp_path):
    state = runtime.RunState(tmp_path / "state.json")

    with pytest.raises(KeyboardInterrupt), state.cell("long job"):
        raise KeyboardInterrupt

    assert state.cells["long job"]["status"] == "interrupted"
    assert not state.ok("long job")


def test_time_left_reserves_a_safety_margin(tmp_path):
    state = runtime.RunState(tmp_path / "state.json", session_limit_s=3600 + runtime.SESSION_SAFETY_S)
    assert 3590 < state.time_left() <= 3600

    state.started -= 10_000
    assert state.time_left() == 0


def test_print_summary_lists_cells_and_artifacts(tmp_path, capsys):
    state = runtime.RunState(tmp_path / "state.json")
    with state.cell("corpus"):
        pass
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "corpus.parquet").write_bytes(b"x" * 2000)

    state.print_summary(artifact_dirs=(tmp_path / "out", tmp_path / "missing"))

    out = capsys.readouterr().out
    assert "corpus" in out and "ok" in out
    assert "corpus.parquet" in out and "(0.00 MB)" in out
    assert "missing: 0 files" in out


def test_run_logged_returns_exit_codes_and_mirrors_output(tmp_path, capsys):
    log = tmp_path / "logs" / "job.log"

    ok = runtime.run_logged([sys.executable, "-c", "print('hello'); print('world')"], log)
    failed = runtime.run_logged([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], log)

    assert (ok, failed) == (0, 3)
    assert log.read_text().split() == ["hello", "world", "boom"]
    assert "hello" in capsys.readouterr().out


def test_run_logged_enforces_the_timeout_without_raising(tmp_path):
    code = runtime.run_logged(
        [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
        tmp_path / "slow.log",
        timeout_s=1,
    )

    assert code == runtime.TIMEOUT_EXIT_CODE
    assert "start" in (tmp_path / "slow.log").read_text()


def test_preflight_collects_every_problem_and_never_prints_secret_values(monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(runtime, "detect_platform", lambda: "local")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_SUPERSECRETVALUE123")

    report = runtime.preflight(
        need_gpu=True, need_cdhit=True, need_hub_write=False, check_internet=False
    )

    assert not report.ok
    assert set(report.failures()) == {"gpu", "cd-hit"}  # all failures reported, not just the first
    assert "wandb" in report.degraded  # missing W&B key degrades softly instead of failing
    out = capsys.readouterr().out
    assert "hf_SUPERSECRETVALUE123" not in out
    assert "secret HF_TOKEN" in out


def test_preflight_rejects_a_read_only_hub_token_when_writes_are_needed(monkeypatch):
    class ReadOnlyApi:
        def __init__(self, token=None):
            pass

        def whoami(self):
            return {"auth": {"accessToken": {"role": "read"}}}

    monkeypatch.setattr("huggingface_hub.HfApi", ReadOnlyApi)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(runtime, "detect_platform", lambda: "local")

    report = runtime.preflight(need_hub_write=True, check_internet=False)

    assert "hub write scope" in report.failures()


def test_preflight_passes_when_the_environment_is_complete(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime, "detect_platform", lambda: "local")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setenv("WANDB_API_KEY", "dummy")

    report = runtime.preflight(need_cdhit=True, check_internet=False)

    assert report.ok and not report.degraded


def test_logged_children_can_print_unicode_even_when_the_parent_uses_a_legacy_encoding(
    monkeypatch, tmp_path
):
    """The training progress cards print box-drawing characters; on Windows a pipe defaults to
    cp1252 and that used to kill the run with UnicodeEncodeError."""
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    code = "print('card \u2500\u2502\u250c done')"

    exit_code = runtime.run_logged([sys.executable, "-c", code], tmp_path / "unicode.log")

    assert exit_code == 0
    assert "─│┌" in (tmp_path / "unicode.log").read_text(encoding="utf-8")


def test_ensure_utf8_output_makes_a_legacy_stream_non_raising(monkeypatch):
    import io

    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", legacy)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    runtime.ensure_utf8_output()
    print("─│ box drawing")  # raises UnicodeEncodeError on a cp1252 stream
    legacy.flush()

    assert legacy.encoding == "utf-8"
