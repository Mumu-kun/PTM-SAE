# PTM Modular SAE Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Architecture: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Toolchain: uv](https://img.shields.io/badge/toolchain-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Code Style: AGENTS.md](https://img.shields.io/badge/code%20style-AGENTS.md-success.svg)](AGENTS.md)

Modular Sparse Autoencoders (SAEs) for Post-Translational Modification (PTM) Mechanistic Interpretability in Protein Language Models (ESM-2). Part of the BUET Undergraduate Thesis (*Member 1 Workstream: Activation Engine & Pipeline Infrastructure*).

---

## Architecture Overview

```
[Raw Sources: UniProt & CPLM] 
            │
            ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 1 & 2: Ingestion, Invariant Check & Partitioning │
 │  • Coordinate invariant: seq[pos - 1] == site.residue  │
 │  • HiGHS MILP 50% Homology Cluster Partitioning:       │
 │    - discovery_train (~70% tokens, 8,894 proteins)     │
 │    - discovery_val   (~10% tokens, 3,240 proteins)     │
 │    - held_out        (~20% tokens, 5,995 proteins)     │
 └────────────────────────────┬───────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 3: ESM-2 Activation Extraction & SafeTensors     │
 │  • Multi-GPU data-parallel extraction (DDP / DP)       │
 │  • Token-level sharding (SafeTensors, 64 MB blocks)    │
 │  • Auxiliary sequence context (mean-pooled embeddings) │
 └────────────────────────────┬───────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 4 & 5: Hugging Face Hub Sync & Readback Sanity   │
 │  • Zero-knowledge token resolution cascade             │
 │  • Exponential backoff upload with SHA-256 integrity   │
 │  • Zero-copy memory-mapped readback verification       │
 └────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
ptm-sae-engine/
├── configs/                   # Model & extraction configurations (dev_8m, colab_650m)
├── data/
│   ├── raw/                   # External raw datasets (UniProt Swiss-Prot, CPLM 4.0)
│   ├── processed/             # Harmonized proteins, PTM sites, split manifests
│   └── acquisition_manifest.json  # Cryptographic SHA-256 provenance tracking
├── docs/
│   ├── adr/                   # Architecture Decision Records (ADR 0001, ADR 0002)
│   └── research/              # Literature benchmarks, stratification, and PTM selection
├── notebooks/
│   └── kaggle_pipeline.ipynb   # Zero-setup Kaggle corpus-build + GPU extraction notebook
├── src/ptm_sae/
│   ├── config.py              # Strict dataclass configurations & YAML parsing
│   ├── data/                  # Ingestion adapters, invariant checking, MILP splitting
│   ├── extraction/            # ESM-2 extractor, SafeTensors sharder, HF sync
│   └── pipeline.py            # Unified end-to-end lifecycle runner
└── tests/                     # Comprehensive test suite (100% passing)
```

---

## Dataset Ingestion & 3-Way Homology Partition

1. **UniProtKB/Swiss-Prot Human Proteome**: Reviewed canonical sequences ($L \le 1022$ AA, 18,129 proteins, ~7.47M tokens).
2. **CPLM 4.0 Human PTMs**: 298,634 experimental/curated lysine modification events (ubiquitin, acetylation, succinylation, etc.).
3. **UniProt Curated Features**: Glycosylation, phosphorylation, methylation, and lipidation features.
4. **Sequence Invariant Enforcement**: Every site coordinate is checked against the canonical FASTA sequence (`sequence[pos - 1] == site.residue`). Isoform drift or coordinate mismatches are logged to `mismatch_audit.tsv`.
5. **Joint 3-Way Cluster Partition** (`ptm_sae.data.splitting`, details in `docs/thesis-meeting-split-strategy.md`): CD-HIT@40% clusters are assigned whole, by one MILP + local-search solve, so the partitions match on tokens, sites per PTM type, residues per stratum, cluster-size mix and labelled-protein share; a cd-hit-2d audit then merges any cluster pair still above 40% identity across a boundary.
   - `discovery_train` (~70% of tokens): unsupervised SAE dictionary learning.
   - `discovery_val` (~10% of tokens): SAE hyperparameter tuning, early stopping and the collapse canaries.
   - `held_out` (~20% of tokens): strictly reserved for non-homologous evaluation.
   - Re-split an already-built corpus without re-clustering: `uv run python -m ptm_sae.corpus.pipeline --resplit` (add `--skip-audit` where `cd-hit-2d` is unavailable).

---

## Quickstart

### 1. Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/your-org/ptm-sae-engine.git
cd ptm-sae-engine

# Sync dependencies in virtual environment
uv sync --extra dev
```

### 2. Run Test Suite

```bash
uv run pytest tests/ -v
```

### 3. Run Pipeline Locally (Sample Mode)

```bash
# Fast sanity run on bundled sample.fasta and ESM-2 8M
uv run python -m ptm_sae.pipeline --config configs/dev_8m.yaml --sample-only
```

### 4. Run Full Lifecycle (Automatic Tier 1 Acquisition)

```bash
# Automatically downloads Swiss-Prot human proteome + CPLM 4.0 + UniProt features,
# harmonizes datasets, generates 3-way split, and extracts activations:
uv run python -m ptm_sae.pipeline \
    --config configs/colab_650m.yaml \
    --remote-repo "your-hf-username/ptm-activations"
```

---

## Running Remotely (GPU box, Kaggle, Colab)

**Which platform:** a run uses **one** platform, never both. Prefer the GPU box; use Kaggle only when the box is unavailable (Colab only if you need a GPU Kaggle cannot offer). The notebooks and CLIs behave identically on all of them, so switching is a matter of where you start the run. Runs do not hand off mid-way unless checkpoints are pushed to the Hub; otherwise restart on the other platform.

The same code and YAML configs run everywhere. Code lives in git (`main`); data and checkpoints live under a **data root** that relative config paths hang off: `$PTM_SAE_DATA_ROOT`, else `/kaggle/working` on Kaggle, else the repo root.

**Kaggle / Colab:** open [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb) (corpus + activation extraction) or [notebooks/train_sae.ipynb](notebooks/train_sae.ipynb) (SAE training). The first cells clone the repo, install the dependencies declared in `pyproject.toml`, and discover secrets (`HF_TOKEN`, `WANDB_API_KEY`, `GH_TOKEN`) from the platform's secret store.

**Persistent Linux GPU box:**

```bash
git clone https://github.com/Mumu-kun/PTM-SAE.git && cd PTM-SAE   # any folder name works
uv sync --extra notebook                      # locked env; keeps the CUDA torch build
export PTM_SAE_DATA_ROOT=/path/to/big/disk    # optional: where cache/, checkpoints/ go

# Pre-download over the (possibly slow) link before training; resumable and idempotent.
# Activations come from ptm-sae-dataset, corpus tables from ptm-sae-corpus (both set in the YAML)
uv run python -m ptm_sae.data.sync --config configs/train_topk_baseline.yaml
uv run python -m ptm_sae.data.sync --config ... --verify   # re-hash local files, repair bad ones
uv run python -m ptm_sae.data.sync --config ... --force    # re-download everything selected

# Train headless (survives a dropped connection) ...
tmux new -s sae
uv run python -m ptm_sae.training.train --config configs/train_topk_baseline.yaml --set total_steps=2000

# ... or drive it from a notebook on your PC through an SSH tunnel
uv run jupyter lab --no-browser --port 8888
ssh -C -o ServerAliveInterval=30 -L 8888:localhost:8888 user@remote   # on your PC
```

The notebooks fast-forward an existing checkout (`git pull --ff-only --autostash`) and never force-reset it; restart the kernel after a pull that changed `src/`.

### Notebooks that survive silly errors

Both notebooks are built so one failing cell cannot throw away a long run (the point on Kaggle, where an unhandled error ends the version):

- Every stage runs inside `with state.cell("name"):` ([ptm_sae.runtime](src/ptm_sae/runtime.py)). A failure is printed, recorded in `run_state.json` (data root) and the notebook continues; cells that depend on it skip with a reason.
- Long jobs (corpus build, extraction, training) run as logged subprocesses (`logs/*.log` in the data root); their exit code is checked, never raised, so a crash or out-of-memory kill cannot take the notebook or its outputs down.
- A preflight cell reports every environment problem at once (GPU, disk, internet, secrets by name, `cd-hit`) and degrades softly where safe (no `WANDB_API_KEY` -> W&B off).
- The last cell always prints which cells completed and which files can be salvaged, even after failures.
- `PTM_SAE_MODE=smoke` runs the whole notebook on tiny/mock data in minutes; `real` is the full run. `PTM_SAE_PULL=0` stops the setup cell from fast-forwarding the checkout. Verify a notebook in smoke mode on the platform you will use before a long run.
- The corpus notebook publishes to the Hub only after a verification cell passes (balance, homology audit, headline bars) and only when `PUBLISH` is on.

### Concurrent runs: sweeps and ablations

An SAE needs far less than a GPU, so several training runs can share one card; the limits are CPU/data loading, RAM and, on Kaggle, the weekly GPU quota. Describe a sweep in YAML ([sweeps/example.yaml](sweeps/example.yaml)) and launch it:

```bash
uv run python -m ptm_sae.training.sweep --spec sweeps/example.yaml --dry-run      # preview the commands
uv run python -m ptm_sae.training.sweep --spec sweeps/example.yaml --max-parallel 3 [--gpus 0,1]
# choose --max-parallel from a measurement, not a guess:
uv run python -m ptm_sae.training.sweep --benchmark --config configs/train_topk_baseline.yaml --levels 1,2,4
```

Each run gets its own checkpoint directory and W&B name/group/tags; a failed run is recorded and the others continue; re-launching resumes (finished runs skipped, interrupted ones continue from `latest/`); the end of the run prints a table of status, val MSE, explained variance, dead-latent fraction and throughput per run.

---

## Coding Standards

This repository strictly adheres to [AGENTS.md](AGENTS.md):
- **Locality**: Cohesive procedures are kept in single linear pipelines without artificial subfunctions.
- **Flat Indentation**: Early guard clauses keep the primary execution path at indentation level 1.
- **Table-Driven Logic**: Lookups use compact, aligned tuples and mappings.
- **Verification First**: Every commit is verified against the full test suite (`uv run pytest tests/ -v`).
