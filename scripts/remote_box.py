"""Thin wrapper for the shared GPU box: develop here, run there.

The box is reached through the tunnel app (local 127.0.0.1:2222 -> box:22) with key auth, so the
tunnel must be running. Nothing outside ~/ptm-sae-engine, ~/.config/ptm-sae and one Jupyter kernel
spec is ever touched on the box. Secrets: keep them in the repo's gitignored .env; `sync` carries it to
the box (mode 600) and `exec` / the Jupyter server load it into the environment, where the project's
resolve_secret() finds them. After editing .env, `kernel down` + `kernel up` so the server picks it up.

    uv run python scripts/remote_box.py sync [--dry-run] [--restart]   # push code; refresh env + kernels if deps changed
    uv run python scripts/remote_box.py setup                          # uv sync on the box + register the kernel
    uv run python scripts/remote_box.py kernel up|down                 # Jupyter server (tmux, idle-culled) + local port forward
    uv run python scripts/remote_box.py kernel list|restart [NOTEBOOK|ID]  # which kernels run what; restart one or all
    uv run python scripts/remote_box.py exec [--tmux NAME] -- <command>  # run anything in the repo dir on the box
    uv run python scripts/remote_box.py pull [PATH]                    # fetch PATH (project-relative, default runs/) from the box

`exec --tmux` runs are guarded by default, because the box is shared: a pre-flight refuses to start when the GPU or RAM
is too full (--wait MIN queues instead), and the run is wrapped to yield (RAM/CPU ceilings that only this run can hit,
lowest priority, first in line for the OOM killer). --gpu-need, --ram, --cpus tune it; --no-guard turns it off.

In VS Code pick "Select Kernel -> Existing Jupyter Server" and paste the URL printed by `kernel up`:
notebooks stay local, cells execute on the box. The server listens on the box's 127.0.0.1 only and is
reached through an ssh -L forward, so no new port is exposed on a machine that is not ours.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".scratch" / "remote_box.json"

HOST = "mdsr@127.0.0.1"
SSH_OPTS = ["-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=40", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
REMOTE_DIR = "ptm-sae-engine"  # relative to the box's $HOME
CONFIG = ".config/ptm-sae"  # token, env file, deps hash; relative to $HOME
KERNEL, KERNEL_TITLE = "ptm-sae-engine", "PTM SAE Engine"
SETUP_SESSION = "ptm-setup"
JUPYTER_PORT, TMUX_SESSION = 38617, "ptm-jupyter"  # same port on both ends; unusual so it cannot clash with the box's other users
PATH_PREFIX = 'export PATH="$HOME/.local/bin:$PATH"; '
LOAD_ENV = "set -a; . <(sed 's/[[:cntrl:]]$//' .env 2>/dev/null); set +a;"  # CRLF-safe: .env is edited on Windows

# Shared-box etiquette for guarded runs: defaults for what a run may take, and what is always left for the others.
GPU_NEED_GB, GPU_MARGIN_GB = 3.0, 2.0
RAM_CAP_GB, RAM_MARGIN_GB = 16.0, 8.0
CPUS, WAIT_POLL_S, CPU_BUSY = 8, 30, 0.7  # CPU is only warned about: nice 19 already makes a run yield
# One ssh call that reads everything the gate needs.
GATE_PROBE = (
    "echo gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1);"
    " echo ram_avail_mib=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo);"
    " echo load1=$(cut -d' ' -f1 /proc/loadavg); echo threads=$(nproc);"
    " echo gpu_apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | paste -sd';')"
)

# Machine-local, regenerable or personal (grant paperwork, credential files other than .env); they stay untouched on the box (no --delete-excluded).
SYNC_EXCLUDES = [
    ".git/", ".venv/", "__pycache__/", ".pytest_cache/", ".ruff_cache/", ".scratch/", ".ipynb_checkpoints/",
    "uv.lock", ".secrets*", "kaggle.json", "*.pem", "*.key", ".claude/", ".verify_state.json", "scratch.md", "proposal/", "wandb/", "runs/", "cache/", "old/", "checkpoints/", "data/raw/", "data/processed/", "data/shards/",
    "*.safetensors", "*.pt", "*.parquet", "*.log",
]

# Runs inside the box's project venv; kept as plain text so it can be piped over ssh.
RESTART_KERNELS = """
import json, os, sys, urllib.request
port, mode, config, target = sys.argv[1:5]
token = open(os.path.expanduser(f"~/{config}/jupyter_token")).read().strip()
def call(path, method="GET"):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/{path}", method=method, headers={"Authorization": "token " + token})
    return json.load(urllib.request.urlopen(request))
try:
    kernels, sessions = call("kernels"), call("sessions")
except OSError:
    print("no Jupyter server is running (kernel up)")
    sys.exit(0)
paths = {session["kernel"]["id"]: session["path"] for session in sessions}
chosen = [k for k in kernels if not target or target in paths.get(k["id"], "") or k["id"].startswith(target)]
if mode == "busy-check":
    busy = [paths.get(k["id"], k["id"][:8]) for k in kernels if k["execution_state"] == "busy"]
    print("busy kernels:", ", ".join(busy) if busy else "none")
    sys.exit(3 if busy else 0)
for kernel in chosen:
    label = f'{kernel["id"][:8]} {kernel["execution_state"]:5} {paths.get(kernel["id"], "(no notebook)")}'
    if mode == "list":
        print(label)
    elif kernel["execution_state"] == "busy" and mode != "force":
        print("skipped busy kernel:", label, "(--force to restart anyway)")
    else:
        call("kernels/" + kernel["id"] + "/restart", "POST")
        print("restarted kernel:", label)
print(len(chosen), "of", len(kernels), "kernel(s)", "matched" if target else "")
"""


def msys_path(path) -> str:
    """Windows path -> the /e/dir form that the msys rsync expects (identity elsewhere)."""
    if os.name != "nt":
        return str(path)
    path = Path(path).resolve()
    return f"/{path.drive.rstrip(':').lower()}{path.as_posix()[2:]}"


def ssh_exe() -> str:
    """Windows OpenSSH by full path: the scoop rsync cannot see ssh on PATH."""
    windows = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32/OpenSSH/ssh.exe"
    return str(windows) if windows.exists() else shutil.which("ssh") or "ssh"


def deps_hash(root: Path) -> str:
    """What decides whether the box's environment is stale. Only pyproject.toml: the box resolves its own
    uv.lock (the committed one pins a PyPI mirror the box cannot reach), so the local lock is irrelevant."""
    return hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()


def rsync_args(source: str, dest: str, *extra: str) -> list[str]:
    return [
        shutil.which("rsync") or "rsync", "-az", "--human-readable", "--itemize-changes",
        "-e", " ".join([msys_path(ssh_exe()), *SSH_OPTS]), *extra, source, dest,
    ]


def pull_dest(root: Path, path: str) -> Path:
    """rsync puts the fetched entry inside its destination, so that is the parent of what was asked for."""
    asked = Path(path)
    if asked.anchor or ".." in asked.parts:  # anchor: also catches /x and C:x, which Windows pathlib calls relative
        raise ValueError(f"pull takes a path inside the project, not {path!r}")
    return root / asked.parent


def parse_probe(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def shortfalls(probe: dict[str, str], gpu_need_gb: float, ram_gb: float) -> list[str]:
    """What the box lacks for a run needing this much on top of the margin we always leave for the others."""
    wants = [
        ("GPU memory", int(probe.get("gpu_free_mib") or 0), gpu_need_gb, GPU_MARGIN_GB),
        ("RAM", int(probe.get("ram_avail_mib") or 0), ram_gb, RAM_MARGIN_GB),
    ]
    return [
        f"{name}: {free / 1024:.1f} GB free, need {need + margin:g} GB ({need:g} for the run + {margin:g} left for others)"
        for name, free, need, margin in wants
        if need > 0 and free < (need + margin) * 1024
    ]


def cpu_warning(probe: dict[str, str]) -> str | None:
    load, threads = float(probe.get("load1") or 0), int(probe.get("threads") or 1)
    if load / threads <= CPU_BUSY:
        return None
    return f"CPU is busy (load {load:.1f} on {threads} threads); the run will run at the lowest priority"


def wait_loop(gpu_need_mib: int, ram_need_mib: int, limit_s: int) -> str:
    """Shell that blocks until the box has room (same numbers as the gate), or exits 75 after limit_s. It runs
    inside the tmux job, so the PC can be off while a run is queued."""
    clauses = []
    if gpu_need_mib:
        clauses.append(f'[ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge {gpu_need_mib} ]')
    if ram_need_mib:
        clauses.append(f"[ \"$(awk '/MemAvailable/ {{print int($2/1024)}}' /proc/meminfo)\" -ge {ram_need_mib} ]")
    if not clauses:
        return ""
    return (
        f"t=0; until {' && '.join(clauses)}; do"
        f" [ $t -ge {limit_s} ] && {{ echo '[gave up: the box never had enough free GPU memory / RAM]'; exit 75; }};"
        f" sleep {WAIT_POLL_S}; t=$((t+{WAIT_POLL_S})); done;"
    )


def polite(command: str, cpus: int, ram_gb: float) -> str:
    """Wrap `command` so it yields to the box's other users. The cgroup scope gives kernel-enforced RAM (no swap)
    and CPU ceilings that only this run can hit; nice/ionice give it the lowest priority; choom makes it the OOM
    killer's first choice; thread counts are capped. Without a user cgroup manager (e.g. no desktop session) it
    still runs, minus the ceilings, and says so."""
    user_bus = "XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus"
    env = f"OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    lowest = f"nice -n 19 ionice -c3 choom -n 500 -- bash -c {shlex.quote(command)}"
    scope = f"systemd-run --user --scope -q -p MemoryMax={ram_gb:g}G -p MemorySwapMax=0 -p CPUQuota={cpus * 100}% --"
    return (
        f"if env {user_bus} systemd-run --user --scope -q true 2>/dev/null; then env {user_bus} {env} {scope} {lowest};"
        f" else echo '[warning: no user cgroup manager, running without RAM/CPU ceilings]'; env {env} {lowest}; fi"
    )


def remote(command: str, *, input: str | None = None, capture: bool = False, check: bool = True, retry: bool = True, tty: bool = False):
    """Run `command` on the box. ssh exits 255 on its own failures (the tunnel's sshd sometimes stalls on
    the banner), so only that is retried; the remote command's own exit codes pass through. Pass
    retry=False for commands that must never run twice: a connection that drops mid-run also exits 255.
    tty=True allocates a terminal, for interactive programs."""
    args = [ssh_exe(), *SSH_OPTS, *(["-t"] if tty else []), HOST, PATH_PREFIX + command]
    for _ in range(3 if retry else 1):
        done = subprocess.run(args, input=input, text=True, encoding="utf-8", capture_output=capture)  # noqa: S603 -- our own ssh, argv list
        if done.returncode != 255:
            break
    if check and done.returncode:
        sys.exit(done.returncode)
    return done


def port_open() -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", JUPYTER_PORT)) == 0


def jupyter_token() -> str:
    return remote(f"cat {CONFIG}/jupyter_token", capture=True).stdout.strip()


def setup() -> None:
    """uv sync + kernel registration, run in tmux on the box and followed with short polls: a multi-GB install
    outlives any single ssh session (this link drops long ones). Re-running while it is still going just
    re-attaches; the deps hash is written only on success."""
    log = f"$HOME/{CONFIG}/setup.log"
    install = (
        f'{PATH_PREFIX} ( cd $HOME/{REMOTE_DIR} && uv sync --extra notebook --extra dev'
        f" && .venv/bin/python -m ipykernel install --user --name {KERNEL} --display-name {shlex.quote(KERNEL_TITLE)}"
        f" && echo {deps_hash(ROOT)} > $HOME/{CONFIG}/deps_hash ) > {log} 2>&1; echo \"[exit $?]\" >> {log}"
    )
    remote(
        f"mkdir -p {CONFIG}; tmux has-session -t {SETUP_SESSION} 2>/dev/null"
        f" || {{ rm -f {log}; tmux new-session -d -s {SETUP_SESSION} {shlex.quote(install)}; }}"
    )

    seen = 0
    while True:
        time.sleep(15)
        done = remote(
            f"tail -c +{seen + 1} {log} 2>/dev/null; tmux has-session -t {SETUP_SESSION} 2>/dev/null || echo '[exit 1: install session vanished]'",
            capture=True,
            check=False,
        )
        if done.returncode == 255:
            continue  # tunnel hiccup: the install keeps running on the box, ask again
        seen += len(done.stdout.encode())
        print(done.stdout, end="", flush=True)
        if "[exit 0]" in done.stdout:
            return
        if "[exit " in done.stdout:
            sys.exit("setup failed on the box; the log is above (also ~/.config/ptm-sae/setup.log)")


def restart_kernels(mode: str = "restart", target: str = "") -> None:
    """mode: list | restart | force | busy-check (exit 3 if any kernel is busy). target: substring of the
    notebook path or a kernel id prefix; empty = all."""
    return remote(
        f"cd {REMOTE_DIR} && .venv/bin/python - {JUPYTER_PORT} {mode} {CONFIG} {shlex.quote(target)}",
        input=RESTART_KERNELS,
        check=False,
    )


def forward_alive(state: dict) -> bool:
    """Is the recorded forward still our ssh? A bare PID is only a hint: after a reboot it may belong to anything."""
    pid = state.get("forward_pid")
    if not pid:
        return False
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True)  # noqa: S603, S607 -- fixed arguments
        return "ssh.exe" in listing.stdout.lower()
    return "ssh" in subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True).stdout  # noqa: S603, S607 -- fixed arguments


def stop_forward() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if forward_alive(state):
        os.kill(state["forward_pid"], signal.SIGTERM)
    STATE.unlink(missing_ok=True)


def kernel_up(idle_kernel: int, idle_server: int) -> None:
    remote(
        f"umask 077; mkdir -p {CONFIG}; [ -s {CONFIG}/jupyter_token ]"
        f" || python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > {CONFIG}/jupyter_token"
    )

    # 1. Server in tmux so it survives disconnects; the token travels in the environment, not in argv
    # (argv is visible to the other users of this box). Idle kernels (they hold VRAM) and then the idle
    # server stop on their own, so a forgotten session on a shared box cleans itself up; busy kernels are never culled.
    server = (
        f"cd $HOME/{REMOTE_DIR} && {LOAD_ENV}"
        f" export JUPYTER_TOKEN=$(cat $HOME/{CONFIG}/jupyter_token);"
        f" exec .venv/bin/jupyter lab --no-browser --ServerApp.ip=127.0.0.1 --ServerApp.port={JUPYTER_PORT}"
        f" --ServerApp.port_retries=0 --ServerApp.root_dir=$HOME/{REMOTE_DIR}"
        f" --MappingKernelManager.cull_idle_timeout={idle_kernel} --MappingKernelManager.cull_interval=60"
        f" --MappingKernelManager.cull_connected=True --ServerApp.shutdown_no_activity_timeout={idle_server}"
    )
    remote(f"tmux has-session -t {TMUX_SESSION} 2>/dev/null || tmux new-session -d -s {TMUX_SESSION} {shlex.quote(server)}")

    # 2. Local forward, detached so it outlives this script
    started = None
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if port_open() and not forward_alive(state):
        sys.exit(f"local port {JUPYTER_PORT} is taken by something that is not our forward; change JUPYTER_PORT")
    if not port_open():
        forward = [ssh_exe(), *SSH_OPTS, "-o", "ExitOnForwardFailure=yes", "-N", "-L", f"{JUPYTER_PORT}:127.0.0.1:{JUPYTER_PORT}", HOST]
        detach = {"creationflags": 0x00000008 | 0x00000200 | 0x08000000} if os.name == "nt" else {"start_new_session": True}
        started = subprocess.Popen(forward, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach)  # noqa: S603 -- our own ssh, argv list
        try:
            STATE.parent.mkdir(exist_ok=True)
            STATE.write_text(json.dumps({"forward_pid": started.pid}))
        except BaseException:
            started.terminate()  # never leave a forward nobody has a record of
            raise

    # 3. Wait until the server answers through the forward
    url, token = f"http://127.0.0.1:{JUPYTER_PORT}", jupyter_token()
    for _ in range(60):
        try:
            request = urllib.request.Request(f"{url}/api/status", headers={"Authorization": f"token {token}"})  # noqa: S310 -- fixed http://127.0.0.1 URL
            urllib.request.urlopen(request, timeout=3).close()  # noqa: S310 -- fixed http://127.0.0.1 URL
            break
        except OSError:
            time.sleep(1)
    else:
        if started:
            stop_forward()  # a forward to nothing is useless; the tmux session stays so the failure can be read
        sys.exit(f"server not answering on {url}; inspect: ssh -p 2222 {HOST} -t tmux attach -t {TMUX_SESSION} (kernel down removes it)")
    print(f"Jupyter server: {url}/?token={token}\nkernel spec to pick: {KERNEL_TITLE}")
    print(f"idle kernels stop after {idle_kernel}s, the idle server after {idle_server}s (0 = never)")


def kernel_down(force: bool) -> None:
    # Stopping the server kills its kernels, and with them any cell that is still computing.
    if restart_kernels("busy-check").returncode == 3 and not force:
        sys.exit("refusing to stop the server while kernels are busy (--force to stop anyway)")
    stop_forward()
    remote(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; true")
    print("server and forward stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="rsync the repo to the box")
    sync.add_argument("--dry-run", action="store_true", help="list what would change, touch nothing")
    sync.add_argument("--restart", action="store_true", help="restart idle kernels even if only code changed")
    sub.add_parser("setup", help="uv sync on the box and register the kernel spec")
    kernel = sub.add_parser("kernel", help="Jupyter server on the box plus local port forward")
    kernel.add_argument("action", choices=["up", "down", "restart", "list"])
    kernel.add_argument("target", nargs="?", default="", help="restart/list: notebook path substring or kernel id prefix (default: all)")
    kernel.add_argument("--force", action="store_true", help="restart / down: also act on busy kernels")
    kernel.add_argument("--idle-kernel", type=int, default=3600, help="up: stop kernels idle this many seconds (0 = never)")
    kernel.add_argument("--idle-server", type=int, default=7200, help="up: stop the server after this many idle seconds with no kernels (0 = never)")
    run = sub.add_parser("exec", help="run a command in the repo dir on the box, with the secrets loaded")
    run.add_argument("--tmux", metavar="NAME", help="detach into a tmux session, logging to runs/logs/NAME.log")
    run.add_argument("--gpu-need", type=float, metavar="GB", help=f"VRAM the run needs (default {GPU_NEED_GB:g}; 0 = no GPU check), plus {GPU_MARGIN_GB:g} GB left for others")
    run.add_argument("--ram", type=float, metavar="GB", help=f"hard RAM ceiling for the run (default {RAM_CAP_GB:g}); {RAM_MARGIN_GB:g} GB more must be available")
    run.add_argument("--cpus", type=int, metavar="N", help=f"CPU ceiling in cores and thread cap (default {CPUS})")
    run.add_argument("--wait", type=int, default=0, metavar="MIN", help="queue on the box for up to MIN minutes instead of refusing when it is too full")
    run.add_argument("--no-guard", action="store_true", help="no pre-flight, no ceilings (e.g. for pytest)")
    run.add_argument("--tty", action="store_true", help="allocate a terminal (interactive programs such as `wandb leet`)")
    run.add_argument("words", nargs=argparse.REMAINDER, help="the command: several words keep their quoting, one word is a shell line (pipes, &&)")
    pull = sub.add_parser("pull", help="fetch a project-relative path from the box")
    pull.add_argument("path", nargs="?", default="runs", help="default: runs")
    args = parser.parse_args()

    if args.command == "sync":
        # The destination is fixed: --delete can only ever act inside ~/ptm-sae-engine.
        remote(f"mkdir -p {REMOTE_DIR}")
        # --max-delete: a sync from a wrong or empty checkout must not be able to wipe the project.
        # --no-perms: leave modes on the box alone, so the 600 on .env sticks.
        flags = ["--delete", "--max-delete=100", "--no-perms", *[f"--exclude={pattern}" for pattern in SYNC_EXCLUDES], *(["--dry-run"] if args.dry_run else [])]
        done = subprocess.run(rsync_args(msys_path(ROOT) + "/", f"{HOST}:{REMOTE_DIR}/", *flags), text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE)  # noqa: S603 -- rsync with fixed arguments
        print(done.stdout)
        if done.returncode:
            sys.exit(done.returncode)
        sent = [line.split(maxsplit=1)[1] for line in done.stdout.splitlines() if line.startswith("<f") and " " in line]
        deleted = [line.split(maxsplit=1)[1] for line in done.stdout.splitlines() if line.startswith("*deleting")]
        print(f"{len(sent)} file(s) {'would be ' if args.dry_run else ''}sent, {len(deleted)} {'would be ' if args.dry_run else ''}deleted")
        for path in deleted[:20]:
            print("  deleted:", path)
        if args.dry_run:
            return
        remote(f"chmod 600 {REMOTE_DIR}/.env 2>/dev/null; true")

        # Stale environment -> refresh it and restart kernels (their imports are frozen); code only -> hint.
        stale = remote(f"cat {CONFIG}/deps_hash 2>/dev/null", capture=True, check=False).stdout.strip() != deps_hash(ROOT)
        if stale:
            print("dependencies changed: uv sync on the box")
            setup()
        if stale or (sent and args.restart):
            restart_kernels()
        elif any(path.endswith((".py", ".yaml")) for path in sent):
            print("code changed: running kernels keep their old imports; `kernel restart` (or sync --restart), or use %autoreload 2")
    elif args.command == "setup":
        setup()
    elif args.command == "kernel":
        {"up": lambda: kernel_up(args.idle_kernel, args.idle_server), "down": lambda: kernel_down(args.force), "restart": lambda: restart_kernels("force" if args.force else "restart", args.target), "list": lambda: restart_kernels("list", args.target)}[args.action]()
    elif args.command == "exec":
        words = args.words[1:] if args.words[:1] == ["--"] else args.words
        if not words:
            sys.exit("exec needs a command")
        # One word is a shell line the caller quoted themselves ("ls | wc -l"); several words are an argv,
        # so their quoting is kept (python -c "print(1)").
        command = words[0] if len(words) == 1 else shlex.join(words)

        # Guarded: runs (--tmux) and anything given a resource flag. The gate refuses (or, with --wait, the job
        # queues on the box) when the box lacks room; the command then runs wrapped to yield to the others.
        asked = any(value is not None for value in (args.gpu_need, args.ram, args.cpus))
        wait = ""
        if not args.no_guard and (args.tmux or asked):
            gpu_need = GPU_NEED_GB if args.gpu_need is None else args.gpu_need
            ram, cpus = args.ram or RAM_CAP_GB, args.cpus or CPUS
            probe = parse_probe(remote(GATE_PROBE, capture=True).stdout)
            lacking = shortfalls(probe, gpu_need, ram)
            if warning := cpu_warning(probe):
                print("warning:", warning)
            if lacking and not args.wait:
                sys.exit("not starting: " + "; ".join(lacking) + f"\nGPU holders (pid, MiB): {probe.get('gpu_apps') or 'none'}\n--wait MIN queues it on the box; --no-guard skips this check")
            if args.wait:
                wait = wait_loop(int((gpu_need + GPU_MARGIN_GB) * 1024) if gpu_need > 0 else 0, int((ram + RAM_MARGIN_GB) * 1024), args.wait * 60)
                print("queued: " + "; ".join(lacking) if lacking else "the box has room now")
            command = polite(command, cpus, ram)

        body = f"{PATH_PREFIX} cd $HOME/{REMOTE_DIR} && {LOAD_ENV} {wait} {command}"
        if not args.tmux:
            remote(body, retry=False, tty=args.tty)
            return
        log = f"runs/logs/{args.tmux}.log"
        inner = f"cd $HOME/{REMOTE_DIR} && mkdir -p runs/logs && {{ {body}; }} 2>&1 | tee {log}; echo \"[exit ${{PIPESTATUS[0]}}]\" >> {log}"
        remote(f"tmux new-session -d -s {shlex.quote(args.tmux)} {shlex.quote(inner)}")
        print(f"started tmux session {args.tmux}; log: {REMOTE_DIR}/{log}; attach: ssh -p 2222 {HOST} -t tmux attach -t {args.tmux}")
    else:
        dest = pull_dest(ROOT, args.path)
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(rsync_args(f"{HOST}:{REMOTE_DIR}/{args.path}", msys_path(dest) + "/", "--progress"), check=True)  # noqa: S603 -- rsync with fixed arguments


if __name__ == "__main__":
    main()
