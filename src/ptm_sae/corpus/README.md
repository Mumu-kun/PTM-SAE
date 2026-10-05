# `ptm_sae.corpus`

What is left of the partner's N1 corpus pipeline (`docs/reference/N1.ipynb`) after the build was
archived: the paths/config, the homology-aware partition assignment with its cd-hit-2d audit
(`clustering.py`), and the maintenance entry points in `pipeline.py` (`verify_outputs`, `resplit`,
`upload_to_hf`).

## The corpus build is archived

Acquisition of the raw PTM sources (M1), Swiss-Prot filtering and CD-HIT clustering (M2), the PTM label
cascade (M3), the N1 two-way split and the `--mock` smoke mode were removed from `main`. The published
corpus is the Hub dataset `mustafa-muhaimin/ptm-sae-corpus` (`corpus/` subfolder); everything downstream
(training, extraction, the audit, Member 2's evaluation) reads those files, never the build code.

To rebuild from raw sources (new Swiss-Prot release, another PTM source or type, a label-rule change),
restore the code from the archive branch and expect to patch it, since nothing has exercised it since:

```
git checkout archive/corpus-build -- src/ptm_sae/corpus/
```

## Re-splitting an already-built corpus

```
uv run python -m ptm_sae.corpus.pipeline              # re-split corpus.parquet in CorpusPaths.processed_dir
uv run python -m ptm_sae.corpus.pipeline --skip-audit # same, without the cd-hit-2d audit
```

`resplit` needs no downloads and no clustering: the clusters (`cluster_id`) and labels are already in
`corpus.parquet` / `labels_stratified.parquet`. `verify_outputs` gates a publish on the leak fraction,
partition balance and the per-type headline bar.

## `cd-hit-2d` -- required system binary (for the audit only)

The homology audit shells out to `cd-hit-2d`. It is a compiled binary, not a Python package: there is no
`uv add` for it. Install one of:

```
sudo apt-get install cd-hit        # Debian/Ubuntu
conda install -c bioconda cd-hit   # any platform
brew install cd-hit                # macOS
```

or build from https://github.com/weizhongli/cdhit. `clustering.check_cdhit_available()` raises a clear
`RuntimeError` if it is missing; pass `--skip-audit` on machines without it (the skip is recorded in
`split_manifest.json`, and `verify_outputs` then refuses to call the corpus publishable).
