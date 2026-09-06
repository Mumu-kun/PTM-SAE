# 0001: Sharded SafeTensors Caching with Residue-Only Index Alignment

We decided to cache intermediate ESM-2 representations as sharded SafeTensors files indexed by a JSON manifest, explicitly stripping `<cls>`, `<eos>`, and `<pad>` delimiter tokens during forward extraction.

## Context
Full-scale Layer 24 caching across ~20,400 reviewed human Swiss-Prot sequences generates ~25.6 GB of uncompressed FP16 activations (1,280 dimensions per residue). Kaggle and Google Colab Pro+ runtimes have strict RAM limits, ephemeral filesystems, and 12-hour session timeouts. Saving monolithic files causes Out-Of-Memory (OOM) errors, while per-protein files choke cloud storage I/O. Furthermore, retaining `<cls>` and `<eos>` tokens pollutes SAE dictionary capacity with delimiter-detecting latents and creates off-by-one indexing errors against biological PTM ground-truth annotations (e.g., dbPTM, Swiss-Prot).

## Decision
1. **Format**: Pack activations into contiguous 2D SafeTensors files of ~500 MB each (`residues_shard_XXXX.safetensors`).
2. **Biological Index Alignment**: Strip `<cls>`, `<eos>`, and `<pad>` tokens during model hook extraction. For sequence length $L$, the saved tensor has shape `(L, hidden_dim)`, guaranteeing that 0-indexed row $i$ corresponds directly to 1-indexed biological residue $i + 1$.
3. **Session Resumption**: Record every completed shard in `manifest.json` with commit status. Restarting extraction skips committed sequences automatically.
4. **Global Auxiliary Representations**: Extraction always computes and caches the mean-pooled sequence representation in a separate SafeTensors file (`mean_pooled_embeddings.safetensors`, ~52 MB) alongside the contiguous residue shards. Downstream training and probing phases can optionally condition on or gate with this context.

## Consequences
- Fast zero-copy memory-mapped (`mmap`) reads during downstream SAE mini-batch sampling.
- Guaranteed zero off-by-one coordinate errors when evaluating against PTM sites.
- Resilience against Colab/Kaggle notebook timeouts and preemptions.
