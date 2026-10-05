# Train box usage

Develop on this PC, run the heavy work (GPU, training, result analysis) on the lab box. Everything goes
through one wrapper, `scripts/remote_box.py`, which is a thin layer over ssh, rsync and Jupyter's REST API.

```
uv run python scripts/remote_box.py <command>
```

## What the box is

- Reached through the tunnel app (local `127.0.0.1:2222` to the box's port 22), key auth, user `mdsr`.
  **The tunnel app must be running.** A "timed out during banner exchange" error is the tunnel being slow:
  the script retries three times by itself, so just run the command again.
- It is a **shared machine and a shared account**: someone else runs notebooks as `mdsr` (`~/ptm-env`,
  `~/Mesbah - PTMs`), another user trains on the same GPU (RTX 5060 Ti, 16 GB). Do not touch their files.
  The script only ever writes to `~/ptm-sae-engine`, `~/.config/ptm-sae` and one Jupyter kernel spec,
  plus `uv` in `~/.local/bin` and its cache.
- Check the GPU before a long run: `exec -- nvidia-smi`.

## One-time setup

1. Copy `.env.example` to `.env` and fill it in: `HF_TOKEN`, `WANDB_API_KEY`, and `KAGGLE_USERNAME` /
   `KAGGLE_KEY` (the two fields of `~/.kaggle/kaggle.json`). `.env` is gitignored.
2. `sync`. The first sync also installs the environment on the box (a few GB of PyTorch/CUDA wheels) and
   registers the Jupyter kernel `ptm-sae-engine` ("PTM SAE Engine"). The install runs in tmux on the box and
   the script follows its log with short polls, so a dropped connection does not kill it: re-run `setup` (or
   `sync`) to re-attach. It prints the box's log if it fails.

   The local `uv.lock` is **not** synced: the committed one pins wheels to a PyPI mirror the box cannot reach.
   The box resolves its own lock (kept in `~/ptm-sae-engine/uv.lock`) from `pyproject.toml`, and rebuilds the
   environment only when `pyproject.toml` changes.

## Daily loop

| Step | Command |
|---|---|
| Push code (and `.env`) to the box | `sync` (add `--dry-run` to preview) |
| Start the notebook server and port forward | `kernel up` (prints a URL with a token) |
| Run a command in the repo on the box | `exec -- uv run pytest tests/ -v` (several words keep their quoting; for pipes or `&&` pass one quoted string: `exec "nvidia-smi | head -5"`) |
| Run a long job that survives disconnects | `exec --tmux smoke -- uv run python -m ptm_sae.training.train ...` |
| Fetch results | `pull` (all of `runs/`) or `pull checkpoints/NAME/metrics.jsonl` (any project-relative path) |
| Open a web tool running on the box | `tunnel up 6006`, then browse `http://127.0.0.1:6006`; `tunnel down` when done |
| Stop the server and forward | `kernel down` (refuses while a kernel is busy; `--force` overrides) |

`sync` mirrors your working tree to `~/ptm-sae-engine` and deletes files there that you deleted locally.
It never touches `.venv`, `runs/`, `wandb/`, `cache/`, `checkpoints/`, `data/raw|processed|shards`, `old/`,
`proposal/` or `.claude/`, so results and the environment on the box survive. It prints what it deleted, and
refuses to delete more than 100 files in one go (a sync from a wrong or empty checkout cannot wipe the
project; raise `--max-delete` in the script for a deliberate big clean-up).

**Anything produced on the box must be written to one of those protected locations**, usually `runs/`, or
the next `sync` deletes it. For example, point an executed notebook copy at `runs/`, not `notebooks/`.

## Runs on a shared box

The GPU, RAM and CPU are shared, so `exec --tmux` runs are **guarded by default**:

- **Pre-flight:** refuses to start when free VRAM is below the run's need (`--gpu-need`, default 3 GB) plus 2 GB left
  for others, or available RAM is below `--ram` (default 16 GB) plus 8 GB. It prints what is free and who holds the GPU.
  `--wait MIN` queues the run on the box instead (it starts by itself once there is room, even if this PC is off;
  gives up with exit 75). A busy CPU only prints a warning.
- **Wrapper:** the run gets a kernel-enforced RAM ceiling (`--ram`, no swap) and CPU ceiling (`--cpus`, default 8) that
  only it can hit, the lowest CPU and I/O priority, first place in the OOM killer's queue, and capped thread counts.
  Without a user cgroup manager on the box it still runs, minus the ceilings, and logs a warning.
- `--no-guard` skips all of it (use for pytest and other non-GPU jobs); `--gpu-need 0` skips only the GPU check.
- Cap the training process itself too: `--set gpu_memory_fraction=0.25` makes it raise OOM at 25% of the card rather
  than squeeze others. Each log tick records `train/vram_reserved_peak_gb` (what neighbours feel),
  `train/vram_allocated_peak_gb` and `train/host_ram_peak_gb`: run a short smoke job first and size the limits from them.

## Viewing runs

wandb's web UI can lag on this box (slow shared uplink; the whole resume bundle is also uploaded as an artifact at every
eval, which is the first thing to try cutting if the lag persists). Faster ways to watch a run:

- **Console card:** loss, mse, L0, speed and peak memory every 50 steps, in the tmux log (`exec --tmux NAME ...`
  then `ssh -p 2222 mdsr@127.0.0.1 -t tmux attach -t NAME`, or read `runs/logs/NAME.log`).
- **wandb LEET**, a live terminal UI that reads wandb's local run files, no upload delay:
  `exec --tty -- uv run wandb leet` (older wandb: `wandb beta leet run`).
- **wandb's web UI after the run:** train with `WANDB_MODE=offline` (no upload during training, so no lag and no
  uplink contention), then `exec -- .venv/bin/wandb sync wandb/offline-run-*` uploads the finished run once to
  wandb.ai. Needs `WANDB_API_KEY` in `.env`. This is how to get the full web UI; the live view is LEET.
- **Any web tool running on the box** (TensorBoard, a small dashboard) opens locally through
  `tunnel up PORT` (it listens on the box's 127.0.0.1; the page appears at `http://127.0.0.1:PORT` here;
  `tunnel list` / `tunnel down [PORT]` to inspect and close). Nothing is exposed on the box's network.
- **`metrics.jsonl`** in the run's `checkpoint_dir`, one row per log event, written with or without wandb:
  `pull checkpoints/NAME/metrics.jsonl`, then plot it locally.

## Notebooks in VS Code

1. `kernel up`, then copy the printed URL.
2. In the notebook: Select Kernel, Existing Jupyter Server, paste the URL, pick **PTM SAE Engine**.
3. Notebooks stay on this PC; cells run on the box's GPU. The project's `find_repo_root()` finds
   `~/ptm-sae-engine` there, so relative data and run paths behave as on any other machine.

## Restarts and resets

| Situation | What to do |
|---|---|
| You edited `src/` or configs | `sync`, then `kernel restart` (the kernel keeps its old imports) |
| `pyproject.toml` changed | `sync` alone: it reinstalls and restarts idle kernels |
| Several notebooks open, restart one | `kernel list`, then `kernel restart train_sae` (path text) or `kernel restart 3fa9c1` (id prefix) |
| A kernel is busy | It is skipped and reported; `--force` restarts it anyway |
| A result must be trustworthy or repeatable | Do not use the live kernel: `exec` a fresh process, or `exec -- uv run jupyter nbconvert --to notebook --execute notebooks/NAME.ipynb` |

Re-running a notebook in the same live kernel reuses the previous run's variables and can hide bugs; restart
first. Two concurrent runs need distinct `--tmux` names and output paths, and they share one 16 GB GPU.

## Secrets

`.env` holds them locally. `sync` copies it to the box (mode 600) and `exec` and the Jupyter server load it
into the environment, where `resolve_secret()` already looks first. `exec` reads it fresh on every call; the
Jupyter server reads it at startup, so after editing `.env` run `sync`, `kernel down`, `kernel up`.
The account is shared, so anyone logged in as `mdsr` can read the file: use tokens you can revoke.

The Kaggle client accepts `KAGGLE_USERNAME` / `KAGGLE_KEY` from the environment, so `kaggle.json` itself is
never copied. One limit: `scripts/kaggle_run.py` reads the username from `~/.kaggle/kaggle.json`, so it
works from this PC but not on the box; push and fetch Kaggle runs from here.

## Clean-up

- Idle kernels (they hold VRAM) stop after 1 hour, and the server after 2 more hours with no kernels, even if
  this PC is off. Change with `kernel up --idle-kernel SECONDS --idle-server SECONDS` (0 = never).
  Busy kernels and `exec --tmux` jobs are never stopped by this.
- `kernel down` stops the server and the local port forward. Logs of `exec --tmux` jobs stay in
  `~/ptm-sae-engine/runs/logs/`.
- Remove the project from the box by hand: `exec` is for the repo dir, so use
  `ssh -p 2222 mdsr@127.0.0.1 "rm -rf ~/ptm-sae-engine ~/.config/ptm-sae"`, and delete the kernel spec
  `~/.local/share/jupyter/kernels/ptm-sae-engine`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Connection timed out during banner exchange` | Tunnel slow or down; retry, then check the tunnel app. |
| `local port 38617 is taken ...` | Something else listens there; change `JUPYTER_PORT` in the script. |
| `server not answering` | Read why: `ssh -p 2222 mdsr@127.0.0.1 -t tmux attach -t ptm-jupyter`. |
| Kernel missing in VS Code | `kernel up` first; the kernel spec exists only after the first `sync`/`setup`. |
| Changed `.env` has no effect in a notebook | The server loads it at startup: `kernel down`, `kernel up`. |
