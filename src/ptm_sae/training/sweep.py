"""Concurrent training runs on one machine: hyperparameter sweeps, ablations, architecture comparisons.

A sweep spec (YAML, see `sweeps/example.yaml`) names a base training config plus either an explicit
list of runs or a grid. The launcher expands it and runs up to N training subprocesses at once (an
SAE is far smaller than the GPU, so several share one card; the real limits are CPU/data loading and
RAM, which `--benchmark` measures). Every run's outcome is recorded in a state file: a failing run
never stops the others, and a re-launch only redoes what is not finished (resuming from `latest/`).

Usage:
    uv run python -m ptm_sae.training.sweep --spec sweeps/example.yaml [--max-parallel 3] [--gpus 0,1]
    uv run python -m ptm_sae.training.sweep --benchmark --config configs/train_topk_baseline.yaml
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ptm_sae import runtime
from ptm_sae.runtime import resolve_data_root

THROUGHPUT = re.compile(r"([\d,]+) tok/s")
VAL_LINE = re.compile(r"val_mse=([\d.eE+-]+)\s+explained_variance=([\d.]+)%")
DEAD_LINE = re.compile(r"dead_latent_fraction=([\d.]+)%")
UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._=-]+")


@dataclass(frozen=True)
class SweepRun:
    name: str
    overrides: dict  # config key (dotted for nested, e.g. "wandb.enabled") -> value


def expand_runs(spec: dict) -> list[SweepRun]:
    """Explicit `runs`, then the cartesian product of `grid`, each optionally crossed with `seeds`."""
    runs = [SweepRun(r["name"], dict(r.get("set", {}))) for r in spec.get("runs", [])]
    grid = spec.get("grid", {})
    if grid:
        keys = list(grid)
        for combo in itertools.product(*(grid[key] for key in keys)):
            name = "_".join(f"{key.split('.')[-1]}{value}" for key, value in zip(keys, combo, strict=True))
            runs.append(SweepRun(name, dict(zip(keys, combo, strict=True))))
    if seeds := spec.get("seeds"):
        runs = [SweepRun(f"{r.name}_s{seed}", {**r.overrides, "seed": seed}) for r in runs for seed in seeds]

    runs = [SweepRun(UNSAFE_NAME.sub("-", r.name), r.overrides) for r in runs]
    if not runs:
        raise ValueError("Sweep spec defines no runs: give `runs:` and/or `grid:`.")
    names = [r.name for r in runs]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"Duplicate run names in sweep spec: {duplicates}")
    return runs


def run_command(spec: dict, run: SweepRun, sweep_dir: Path, data_root: Path) -> list[str]:
    """The training CLI call for one run: its own checkpoint dir, W&B name/group/tags, and the run's
    overrides. Values are passed as JSON, which the config's YAML parser reads back as-is."""
    checkpoint_dir = sweep_dir / run.name
    overrides = {
        **run.overrides,
        "checkpoint_dir": str(checkpoint_dir),
        "wandb.run_name": f"{spec['group']}_{run.name}",
        "wandb.group": spec["group"],
        "wandb.tags": [*spec.get("tags", []), "sweep", spec["group"]],
    }
    cmd = [
        sys.executable, "-u", "-m", "ptm_sae.training.train",
        "--config", spec["base_config"], "--data-root", str(data_root),
    ]  # fmt: skip
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={json.dumps(value)}"]
    if (checkpoint_dir / "latest").exists():  # an interrupted earlier attempt: continue it
        cmd += ["--resume-from", str(checkpoint_dir)]
    return cmd


class SweepState:
    """Per-run status persisted after every change, so a killed launcher resumes where it was."""

    def __init__(self, path: Path):
        self.path = path
        self.runs: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def set(self, name: str, **fields) -> None:
        self.runs.setdefault(name, {}).update(fields)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.runs, indent=2))
        os.replace(tmp, self.path)

    def done(self, name: str, sweep_dir: Path) -> bool:
        return self.runs.get(name, {}).get("status") == "ok" and (sweep_dir / name / "best").exists()


def parse_log(log_path: Path) -> dict:
    """Last validation metrics and recent throughput from a run's log (None where not yet logged)."""
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    text = text.replace("\r", "\n")
    val, dead, tput = VAL_LINE.findall(text), DEAD_LINE.findall(text), THROUGHPUT.findall(text)
    recent = [float(t.replace(",", "")) for t in tput[-3:]]
    return {
        "val_mse": float(val[-1][0]) if val else None,
        "explained_variance_pct": float(val[-1][1]) if val else None,
        "dead_latent_pct": float(dead[-1]) if dead else None,
        "tokens_per_sec": sum(recent) / len(recent) if recent else None,
    }


def format_table(state: SweepState, sweep_dir: Path) -> str:
    cell = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    lines = [f"{'run':36s} {'status':9s} {'sec':>6s} {'val_mse':>9s} {'EV%':>6s} {'dead%':>6s} {'tok/s':>8s}"]
    for name, info in state.runs.items():
        m = parse_log(sweep_dir / f"{name}.log")
        lines.append(
            f"{name:36s} {info.get('status', '?'):9s} {cell(info.get('seconds'), '6.0f'):>6s} "
            f"{cell(m['val_mse'], '9.4f'):>9s} {cell(m['explained_variance_pct'], '6.1f'):>6s} "
            f"{cell(m['dead_latent_pct'], '6.1f'):>6s} {cell(m['tokens_per_sec'], '8.0f'):>8s}"
        )
    return "\n".join(lines)


def default_parallel(n_devices: int) -> int:
    """A conservative starting cap per GPU: about half the CPU cores (data loading is the bottleneck
    long before the GPU is), at most 4. `--benchmark` replaces the guess with a measurement."""
    return max(1, min(4, (os.cpu_count() or 2) // 2)) * max(1, n_devices)


def launch(
    spec: dict,
    data_root: str | Path | None = None,
    max_parallel: int | None = None,
    devices: list[int] | None = None,
    timeout_s: float | None = None,
    dry_run: bool = False,
    poll_s: float = 2.0,
) -> SweepState:
    """Runs the sweep and returns its state. Never raises because a run failed: failures are
    recorded as `failed` (non-zero exit) or `timeout` (stopped at `timeout_s`) and the rest go on."""
    data_root = Path(data_root or resolve_data_root())
    sweep_dir = data_root / "sweeps" / spec["group"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    state = SweepState(sweep_dir / "state.json")
    runs = expand_runs(spec)
    max_parallel = max_parallel or spec.get("max_parallel") or default_parallel(len(devices or []))
    pending = [r for r in runs if not state.done(r.name, sweep_dir)]
    print(f"[sweep] {spec['group']}: {len(runs)} runs, {len(runs) - len(pending)} already done, "
          f"max {max_parallel} at once, devices {devices or 'default'}")  # fmt: skip

    if dry_run:
        for run in pending:
            print(" ".join(run_command(spec, run, sweep_dir, data_root)))
        return state

    deadline = time.time() + timeout_s if timeout_s else None
    running: dict[str, tuple[subprocess.Popen, float, int | None]] = {}
    while pending or running:
        out_of_time = deadline is not None and time.time() >= deadline
        while pending and len(running) < max_parallel and not out_of_time:
            run = pending.pop(0)
            used = [d for _, _, d in running.values()]
            device = min(devices, key=used.count) if devices else None
            env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": str(device)} if device is not None else {})}
            with open(sweep_dir / f"{run.name}.log", "a", encoding="utf-8") as log:
                proc = subprocess.Popen(  # noqa: S603 -- our own training CLI, argv list
                    run_command(spec, run, sweep_dir, data_root), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )  # fmt: skip
            running[run.name] = (proc, time.time(), device)
            state.set(run.name, status="running", device=device, started=time.time())
            print(f"[sweep] started {run.name} (pid {proc.pid}, device {device})")

        time.sleep(poll_s)
        for name, (proc, started, _) in list(running.items()):
            code = proc.poll()
            if code is None and not out_of_time:
                continue
            if code is None:
                proc.terminate()
                proc.wait()
                status = "timeout"
            else:
                status = "ok" if code == 0 else "failed"
            state.set(name, status=status, returncode=proc.returncode, finished=time.time(), seconds=round(time.time() - started, 1))
            print(f"[sweep] {name}: {status}")
            del running[name]
        if out_of_time and not running:
            for run in pending:
                state.set(run.name, status="not started")
            break

    print(format_table(state, sweep_dir))
    return state


def benchmark_concurrency(
    base_config: str,
    levels: list[int],
    steps: int,
    overrides: dict,
    data_root: str | Path | None = None,
    devices: list[int] | None = None,
) -> list[dict]:
    """Runs `level` identical short trainings at once for each level and reports per-run and
    aggregate tokens/sec, so the concurrency cap is chosen from a measurement."""
    results = []
    for level in levels:
        spec = {
            "base_config": base_config,
            "group": f"bench_c{level}",
            "runs": [{"name": f"r{i}", "set": {**overrides, "total_steps": steps, "wandb.enabled": False}} for i in range(level)],
            "max_parallel": level,
        }  # fmt: skip
        sweep_dir = Path(data_root or resolve_data_root()) / "sweeps" / spec["group"]
        shutil.rmtree(sweep_dir, ignore_errors=True)  # every level starts from scratch
        state = launch(spec, data_root=data_root, max_parallel=level, devices=devices)
        per_run = [parse_log(sweep_dir / f"{name}.log")["tokens_per_sec"] for name in state.runs]
        ok = [t for t in per_run if t]
        results.append({"concurrent": level, "ok_runs": len(ok), "per_run_tok_s": sum(ok) / len(ok) if ok else None, "aggregate_tok_s": sum(ok)})
    base = results[0]["aggregate_tok_s"] or 1
    print(f"\n{'concurrent':>10s} {'per-run tok/s':>14s} {'aggregate tok/s':>16s} {'speedup':>8s}")
    for r in results:
        per_run = "-" if r["per_run_tok_s"] is None else f"{r['per_run_tok_s']:.0f}"
        print(f"{r['concurrent']:>10d} {per_run:>14s} {r['aggregate_tok_s']:>16.0f} {r['aggregate_tok_s'] / base:>7.2f}x")
    return results


def pick_parallel(results: list[dict]) -> int:
    """Smallest concurrency whose aggregate tokens/sec is within 5% of the best measured: past that
    point extra runs only compete for CPU and memory."""
    measured = [r for r in results if r["aggregate_tok_s"]]
    if not measured:
        return 1
    best = max(r["aggregate_tok_s"] for r in measured)
    return min(r["concurrent"] for r in measured if r["aggregate_tok_s"] >= 0.95 * best)


def _parse_set(items: list[str]) -> dict:
    return {key: yaml.safe_load(value) for key, _, value in (item.partition("=") for item in items)}


def main() -> None:
    runtime.ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Launch concurrent SAE training runs (sweeps, ablations).")
    parser.add_argument("--spec", type=str, help="Sweep spec YAML (see sweeps/example.yaml)")
    parser.add_argument("--max-parallel", type=int, default=None, help="Concurrent runs (default: spec, else a CPU-based guess)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids to spread runs over, e.g. 0,1")
    parser.add_argument("--timeout-hours", type=float, default=None, help="Stop launching/terminate runs after this long")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    parser.add_argument("--benchmark", action="store_true", help="Measure throughput at several concurrency levels")
    parser.add_argument("--config", type=str, help="With --benchmark: base training config")
    parser.add_argument("--levels", type=str, default="1,2,4", help="With --benchmark: concurrency levels")
    parser.add_argument("--steps", type=int, default=300, help="With --benchmark: steps per run")
    parser.add_argument("--benchmark-out", type=Path, help="With --benchmark: write the results (and the chosen concurrency) as JSON")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="With --benchmark: config override applied to every run")
    args = parser.parse_args()

    devices = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    if args.benchmark:
        if not args.config:
            parser.error("--benchmark needs --config")
        results = benchmark_concurrency(
            args.config, [int(x) for x in args.levels.split(",")], args.steps,
            _parse_set(args.overrides), args.data_root, devices,
        )  # fmt: skip
        chosen = pick_parallel(results)
        print(f"[benchmark] chosen concurrency: {chosen}")
        if args.benchmark_out:
            args.benchmark_out.write_text(json.dumps({"results": results, "chosen": chosen}, indent=2), encoding="utf-8")
        return
    if not args.spec:
        parser.error("give --spec (or --benchmark)")
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    launch(
        spec, args.data_root, args.max_parallel, devices,
        args.timeout_hours * 3600 if args.timeout_hours else None, args.dry_run,
    )  # fmt: skip


if __name__ == "__main__":
    main()
