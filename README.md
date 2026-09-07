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
│   └── kaggle_extraction.ipynb # Zero-setup Kaggle GPU extraction notebook
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
5. **3-Way MILP Cluster Partition**:
   - `discovery_train`: 8,894 proteins (67.95% tokens) — for unsupervised SAE dictionary learning.
   - `discovery_val`: 3,240 proteins (9.72% tokens) — for SAE hyperparameter tuning and Pareto frontier evaluation.
   - `held_out`: 5,995 proteins (22.33% tokens) — strictly reserved for non-homologous zero-shot evaluation.

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

## Cloud Execution (Kaggle & Colab)

For multi-GPU or T4/P100/A100 cloud execution, open [notebooks/kaggle_extraction.ipynb](notebooks/kaggle_extraction.ipynb).
The notebook automatically detects Kaggle environments, configures local scratch storage paths, and handles asynchronous chunk uploading to Hugging Face Hub.

---

## Coding Standards

This repository strictly adheres to [AGENTS.md](AGENTS.md):
- **Locality**: Cohesive procedures are kept in single linear pipelines without artificial subfunctions.
- **Flat Indentation**: Early guard clauses keep the primary execution path at indentation level 1.
- **Table-Driven Logic**: Lookups use compact, aligned tuples and mappings.
- **Verification First**: Every commit is verified against the full test suite (`uv run pytest tests/ -v`).
