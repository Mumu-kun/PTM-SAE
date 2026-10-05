"""scripts/remote_box.py: the parts that need no ssh (paths, rsync arguments, dependency hash, env parsing)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote_box.py"


@pytest.fixture(scope="module")
def remote_box():
    spec = importlib.util.spec_from_file_location("remote_box", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deps_hash_follows_pyproject_only(remote_box, tmp_path):
    # The box resolves its own lock, so the local uv.lock must not mark its environment stale.
    (tmp_path / "pyproject.toml").write_text("a")
    (tmp_path / "uv.lock").write_text("b")
    before = remote_box.deps_hash(tmp_path)

    (tmp_path / "uv.lock").write_text("changed")
    assert remote_box.deps_hash(tmp_path) == before

    (tmp_path / "pyproject.toml").write_text("changed")
    assert remote_box.deps_hash(tmp_path) != before


def test_rsync_args_keep_the_destination_and_extras_in_place(remote_box):
    args = remote_box.rsync_args("/src/", "mdsr@host:dir/", "--delete", "--dry-run")

    assert args[-2:] == ["/src/", "mdsr@host:dir/"]
    assert args.index("--delete") < args.index("/src/")
    assert "-e" in args


def test_sync_protects_box_state_and_carries_only_the_env_secrets_file(remote_box):
    excludes = set(remote_box.SYNC_EXCLUDES)

    # .venv and runs/ on the box must survive a sync that mirrors the working tree.
    assert {".venv/", "runs/", "wandb/", "uv.lock"} <= excludes
    # .env is the one credential file that travels; other credential files never do.
    assert ".env" not in excludes
    assert {"kaggle.json", "*.pem", "*.key"} <= excludes


def test_msys_path_maps_drive_letters_on_windows(remote_box):
    if remote_box.os.name != "nt":
        pytest.skip("Windows-only path mapping")

    assert remote_box.msys_path(Path("E:/a/b")) == "/e/a/b"


def test_remote_retries_only_ssh_failures_and_never_when_told_not_to(remote_box, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return remote_box.subprocess.CompletedProcess(args, 255)

    monkeypatch.setattr(remote_box.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        remote_box.remote("true", retry=False)
    assert len(calls) == 1

    calls.clear()
    with pytest.raises(SystemExit):
        remote_box.remote("true")
    assert len(calls) == 3


def test_shortfalls_leave_a_margin_for_the_others(remote_box):
    # Run needs 3 GB VRAM + 2 GB margin = 5120 MiB, and 16 GB RAM + 8 GB margin = 24576 MiB.
    enough = {"gpu_free_mib": "5120", "ram_avail_mib": "24576"}

    assert remote_box.shortfalls(enough, 3, 16) == []

    short = remote_box.shortfalls({"gpu_free_mib": "5119", "ram_avail_mib": "24575"}, 3, 16)
    assert [message.split(":")[0] for message in short] == ["GPU memory", "RAM"]
    assert "left for others" in short[0]


def test_shortfalls_skip_the_gpu_for_runs_that_need_none(remote_box):
    assert remote_box.shortfalls({"gpu_free_mib": "", "ram_avail_mib": "99999"}, 0, 16) == []


def test_cpu_is_only_a_warning_when_the_box_is_busy(remote_box):
    assert remote_box.cpu_warning({"load1": "10.0", "threads": "28"}) is None
    assert "busy" in remote_box.cpu_warning({"load1": "21.0", "threads": "28"})


def test_parse_probe_reads_key_value_lines(remote_box):
    assert remote_box.parse_probe("gpu_free_mib=1081\nram_avail_mib=52000\ngpu_apps=1 2;3 4\n") == {
        "gpu_free_mib": "1081",
        "ram_avail_mib": "52000",
        "gpu_apps": "1 2;3 4",
    }


def test_polite_sets_ceilings_priority_and_keeps_pipelines_inside(remote_box):
    wrapped = remote_box.polite("ls | wc -l", 8, 16)

    assert "MemoryMax=16G" in wrapped and "MemorySwapMax=0" in wrapped and "CPUQuota=800%" in wrapped
    assert "nice -n 19 ionice -c3 choom -n 500 -- bash -c 'ls | wc -l'" in wrapped
    assert "OMP_NUM_THREADS=8" in wrapped
    assert "running without RAM/CPU ceilings" in wrapped  # fallback when there is no user cgroup manager
    assert "MemoryMax=1.5G" in remote_box.polite("true", 2, 1.5)


def test_wait_loop_uses_the_gate_thresholds_and_gives_up_with_75(remote_box):
    loop = remote_box.wait_loop(5120, 24576, 3600)

    assert "-ge 5120" in loop and "-ge 24576" in loop and "exit 75" in loop
    assert "[ $t -ge 3600 ]" in loop  # the give-up limit
    assert "nvidia-smi" not in remote_box.wait_loop(0, 24576, 60)
    assert remote_box.wait_loop(0, 0, 60) == ""


def test_generated_shell_is_valid_bash(remote_box):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if not bash or subprocess.run([bash, "-c", "echo ok"], capture_output=True, text=True).stdout.strip() != "ok":  # noqa: S603 -- fixed argv
        pytest.skip("no usable local bash")

    for script in (
        remote_box.polite("python -c 'print(1)' | tee x", 4, 8),
        remote_box.wait_loop(5120, 24576, 600),
        remote_box.GATE_PROBE,
        remote_box.LOAD_ENV,
    ):
        assert subprocess.run([bash, "-n", "-c", script], capture_output=True, text=True).returncode == 0, script  # noqa: S603 -- fixed argv


def test_pull_dest_is_the_parent_of_what_is_asked_for_and_stays_in_the_project(remote_box, tmp_path):
    assert remote_box.pull_dest(tmp_path, "runs") == tmp_path
    assert remote_box.pull_dest(tmp_path, "checkpoints/a/metrics.jsonl") == tmp_path / "checkpoints" / "a"

    for outside in ("../x", "/etc/passwd"):
        with pytest.raises(ValueError):
            remote_box.pull_dest(tmp_path, outside)
