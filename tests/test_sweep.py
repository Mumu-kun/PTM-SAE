"""Sweep launcher: grid expansion, concurrency cap, failure isolation, resume, timeouts, benchmark.

The trainer is replaced by tiny real subprocesses (a short Python script per run), so scheduling,
log parsing and process handling are exercised for real while staying offline and fast."""

import json
import sys
import time
from pathlib import Path

import pytest
import yaml

from ptm_sae.training import sweep

# A fake "training run": logs like the real trainer, records when it ran, leaves a best/ checkpoint.
FAKE_TRAIN = """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
ckpt = Path(next(a for a in args if a.startswith("checkpoint_dir=")).split("=", 1)[1].strip('"'))
name = ckpt.name
trace = ckpt.parent / f"trace-{name}.jsonl"  # one file per run: concurrent appends to a shared file lose lines
with open(trace, "a") as f:
    f.write(json.dumps({"name": name, "event": "start", "t": time.time(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
print("1234 tok/s  ETA 1s", flush=True)
time.sleep(float(os.environ.get("FAKE_SECONDS", "0.4")))
print("val_mse=0.5000  explained_variance=61.5%", flush=True)
print("mean_l0=16.0  dead_latent_fraction=2.5%", flush=True)
if "fail" in name:
    sys.exit(3)
(ckpt / "best").mkdir(parents=True, exist_ok=True)
(ckpt / "latest").mkdir(parents=True, exist_ok=True)
with open(trace, "a") as f:
    f.write(json.dumps({"name": name, "event": "end", "t": time.time()}) + "\\n")
"""


@pytest.fixture
def fake_trainer(monkeypatch, tmp_path):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN)
    real = sweep.run_command

    def command(spec, run, sweep_dir, data_root):
        # keep the real command's --set list (checkpoint_dir first among them), swap the program
        args = real(spec, run, sweep_dir, data_root)
        first_set = args.index("--set")
        return [sys.executable, str(script), *args[first_set:]]

    monkeypatch.setattr(sweep, "run_command", command)
    return script


def _events(sweep_dir: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(sweep_dir.glob("trace-*.jsonl")) for line in f.read_text().splitlines()]


def _spec(**extra):
    return {"base_config": "configs/x.yaml", "group": "g", **extra}


def test_expand_runs_combines_runs_grid_and_seeds():
    spec = _spec(
        runs=[{"name": "batchtopk", "set": {"sae_type": "batchtopk"}}],
        grid={"k": [16, 32], "learning_rate": [0.001]},
        seeds=[0, 1],
    )

    runs = sweep.expand_runs(spec)

    assert [r.name for r in runs] == [
        "batchtopk_s0",
        "batchtopk_s1",
        "k16_learning_rate0.001_s0",
        "k16_learning_rate0.001_s1",
        "k32_learning_rate0.001_s0",
        "k32_learning_rate0.001_s1",
    ]
    assert runs[2].overrides == {"k": 16, "learning_rate": 0.001, "seed": 0}


def test_expand_runs_rejects_empty_and_duplicate_specs():
    with pytest.raises(ValueError, match="no runs"):
        sweep.expand_runs(_spec())
    with pytest.raises(ValueError, match="Duplicate"):
        sweep.expand_runs(_spec(runs=[{"name": "a"}, {"name": "a"}]))


def test_run_names_are_made_filesystem_safe():
    runs = sweep.expand_runs(_spec(grid={"wandb.project": ["my proj/1"]}))

    assert runs[0].name == "project" + "my-proj-1"


def test_run_command_gives_each_run_its_own_checkpoint_dir_and_wandb_identity(tmp_path):
    spec = _spec(tags=["ablation"])
    run = sweep.SweepRun("k16", {"k": 16, "wandb.enabled": False})

    cmd = sweep.run_command(spec, run, tmp_path / "sweeps" / "g", tmp_path)

    sets = dict(item.split("=", 1) for item in cmd[cmd.index("--set") :] if "=" in item)
    assert json.loads(sets["k"]) == 16 and json.loads(sets["wandb.enabled"]) is False
    assert json.loads(sets["checkpoint_dir"]) == str(tmp_path / "sweeps" / "g" / "k16")
    assert json.loads(sets["wandb.run_name"]) == "g_k16" and json.loads(sets["wandb.group"]) == "g"
    assert json.loads(sets["wandb.tags"]) == ["ablation", "sweep", "g"]
    assert "--resume-from" not in cmd

    (tmp_path / "sweeps" / "g" / "k16" / "latest").mkdir(parents=True)
    assert "--resume-from" in sweep.run_command(spec, run, tmp_path / "sweeps" / "g", tmp_path)


def test_launch_runs_everything_and_respects_the_concurrency_cap(fake_trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SECONDS", "0.6")
    spec = _spec(grid={"k": [1, 2, 3, 4]})

    state = sweep.launch(spec, tmp_path, max_parallel=2, poll_s=0.05)

    assert {n: i["status"] for n, i in state.runs.items()} == {f"k{k}": "ok" for k in (1, 2, 3, 4)}
    # Concurrency from the launcher's own start/finish records (independent of interpreter start-up time)
    spans = [(i["started"], i["finished"]) for i in state.runs.values()]
    peak = max(sum(start <= t < end for start, end in spans) for t, _ in spans)
    assert peak == 2  # two ran at once, never three


def test_a_failing_run_is_recorded_and_does_not_stop_the_others(fake_trainer, tmp_path, capsys):
    spec = _spec(runs=[{"name": "good1"}, {"name": "fail_me"}, {"name": "good2"}])

    state = sweep.launch(spec, tmp_path, max_parallel=3, poll_s=0.05)

    assert {n: i["status"] for n, i in state.runs.items()} == {"good1": "ok", "fail_me": "failed", "good2": "ok"}
    assert state.runs["fail_me"]["returncode"] == 3
    table = capsys.readouterr().out
    assert "fail_me" in table and "0.5000" in table  # metrics parsed from the logs


def test_relaunch_skips_finished_runs_and_retries_failed_ones(fake_trainer, tmp_path):
    spec = _spec(runs=[{"name": "good"}, {"name": "fail_me"}])
    sweep.launch(spec, tmp_path, max_parallel=2, poll_s=0.05)
    sweep_dir = tmp_path / "sweeps" / "g"
    starts_before = sum(e["event"] == "start" for e in _events(sweep_dir))

    state = sweep.launch(spec, tmp_path, max_parallel=2, poll_s=0.05)

    assert sum(e["event"] == "start" for e in _events(sweep_dir)) == starts_before + 1  # only the failed run restarted
    assert state.runs["good"]["status"] == "ok" and state.runs["fail_me"]["status"] == "failed"


def test_runs_are_spread_over_the_given_gpus(fake_trainer, tmp_path):
    spec = _spec(runs=[{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}])

    sweep.launch(spec, tmp_path, max_parallel=4, devices=[0, 1], poll_s=0.05)

    gpus = sorted(e["gpu"] for e in _events(tmp_path / "sweeps" / "g") if e["event"] == "start")
    assert gpus == ["0", "0", "1", "1"]


def test_timeout_terminates_running_runs_and_marks_the_rest_not_started(fake_trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SECONDS", "30")
    spec = _spec(runs=[{"name": "slow1"}, {"name": "slow2"}, {"name": "never"}])
    started = time.time()

    state = sweep.launch(spec, tmp_path, max_parallel=2, timeout_s=1.5, poll_s=0.1)

    assert time.time() - started < 15
    assert {n: i["status"] for n, i in state.runs.items()} == {
        "slow1": "timeout",
        "slow2": "timeout",
        "never": "not started",
    }


def test_dry_run_prints_commands_without_running_them(fake_trainer, tmp_path, capsys):
    sweep.launch(_spec(grid={"k": [1, 2]}), tmp_path, dry_run=True)

    assert capsys.readouterr().out.count("--set") >= 2
    assert _events(tmp_path / "sweeps" / "g") == []


def test_benchmark_reports_aggregate_throughput_per_concurrency_level(fake_trainer, tmp_path, capsys):
    results = sweep.benchmark_concurrency(
        "configs/x.yaml", levels=[1, 2], steps=10, overrides={}, data_root=tmp_path
    )

    assert [r["concurrent"] for r in results] == [1, 2]
    assert [r["ok_runs"] for r in results] == [1, 2]
    assert results[1]["aggregate_tok_s"] == pytest.approx(2 * 1234)
    assert "speedup" in capsys.readouterr().out


def test_example_spec_expands():
    spec = yaml.safe_load(Path("sweeps/example.yaml").read_text(encoding="utf-8"))

    runs = sweep.expand_runs(spec)

    assert len(runs) == (3 * 2 + 1) * 2  # (grid + the explicit run) x 2 seeds


def test_launcher_overrides_round_trip_through_the_real_training_config(tmp_path):
    from ptm_sae.training.config import SAETrainingConfig

    spec = _spec(base_config="configs/train_topk_baseline.yaml", tags=["ablation"])
    run = sweep.SweepRun("k16", {"k": 16, "learning_rate": 0.001, "wandb.enabled": False})
    cmd = sweep.run_command(spec, run, tmp_path / "sweeps" / "g", tmp_path)
    overrides = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--set"]

    config = SAETrainingConfig.from_yaml(spec["base_config"]).with_overrides(overrides)

    assert (config.k, config.learning_rate, config.wandb.enabled) == (16, 0.001, False)
    assert config.checkpoint_dir == str(tmp_path / "sweeps" / "g" / "k16")  # backslashes survive
    assert (config.wandb.run_name, config.wandb.group) == ("g_k16", "g")
    assert config.wandb.tags == ["ablation", "sweep", "g"]
