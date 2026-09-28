# `ptm_sae.corpus`

Ports the partner's `src/ptm_eval/N1.ipynb` (Swiss-Prot acquisition -> CD-HIT clustering ->
PTM label cascade) into importable modules. See
`proposal/` and the implementation plan for the full design rationale; this file covers only the
one operational dependency this subpackage needs that `uv` cannot install for you.

## `cd-hit` / `cd-hit-2d` -- required system binary

`corpus/clustering.py::run_cdhit` shells out to the `cd-hit` command-line tool. It is a compiled
binary, not a Python package -- there is no `uv add` for it, and the environment running
`corpus/pipeline.py` may not be the same one running SAE training.

Install one of:

```
# Debian/Ubuntu
sudo apt-get install cd-hit

# conda (any platform)
conda install -c bioconda cd-hit

# macOS (Homebrew)
brew install cd-hit
```

Or build from source: https://github.com/weizhongli/cdhit

Verify both binaries are on `PATH` before running M2:

```
cd-hit -h
cd-hit-2d -h
```

`clustering.check_cdhit_available()` raises a clear `RuntimeError` naming these install commands
if either binary is missing, rather than letting `subprocess.run` fail with an opaque
`FileNotFoundError`.

### Invocation pattern

`run_cdhit` invokes it as:

```
cd-hit -i <fasta> -o <out> -c 0.40 -n 2 -M 0 -T 0 -d 0 -l 1
```

The identity threshold (`-c 0.40`) is a fixed, non-configurable value in this codebase (the
user's decision -- see the implementation plan) rather than a tunable; `-n`/`-l` follow CD-HIT's
documented word-size/throw-away-length bands for that threshold.

## Caching / fingerprint caveat (inherited from N1, not silently fixed)

`config.config_fingerprint` hashes **config**, not **code**. If you edit parsing or filtering
*logic* in `acquisition.py`/`clustering.py`/`labels.py` without changing a `Config` field, the
on-disk cache will not detect the change and will happily reuse stale output. Pass `force=True`
(or `python -m ptm_sae.corpus.pipeline --force`) after any logic edit. This is the same limitation
N1's own `FORCE_REBUILD` flag documents -- it is deliberately not "fixed" here (that would be
separate hardening work, out of scope for this port).
