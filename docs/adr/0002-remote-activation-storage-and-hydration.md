# 0002: Remote Activation Storage and Lazy Shard Hydration via Hugging Face Hub

We decided to use Hugging Face Hub (Private Dataset) as the canonical remote storage authority for activation shards, using atomic single-shard uploads during extraction, on-demand local shard hydration (`hf_hub_download`) during SAE training, and strict zero-knowledge secret isolation across Local, Kaggle, and Google Colab environments.

## Context
Full-scale Layer 24 caching of ESM-2 650M across the human proteome yields ~28.9 GB of FP16 SafeTensors activations. The project operates across heterogeneous, ephemeral compute environments (local workstation, Kaggle GPUs, Google Colab Pro+). Local storage lacks easy cross-cloud synchronization; Google Drive imposes disruptive headless download quota bans; and Kaggle Datasets create high friction when executing outside the Kaggle ecosystem. Furthermore, downloading a monolithic 29 GB archive before starting SAE training on ephemeral instances exhausts disk space and introduces excessive startup latency.

## Decision
1. **Authoritative Host & Hierarchy**: Store all activation shards, auxiliary context embeddings, and partition manifests in a single private Hugging Face Dataset repository (`{username}/ptm-activations`). Organize files strictly by model and layer namespace:
   ```
   ptm-activations/
   └── esm2_t33_650M_UR50D/
       └── layer_24/
           ├── manifest.json
           ├── split_manifest.json
           ├── mean_pooled_embeddings.safetensors
           └── shards/
               ├── shard_0000.safetensors
               └── shard_0001.safetensors
   ```
2. **Bi-directional Cloud Extraction with Resumption**: Extraction workers on Kaggle/Colab/Local query the remote manifest at startup, skip already committed UniProt IDs, and upload completed `shard_XXXX.safetensors` atomically via the Hugging Face REST API (`HfApi.upload_file`) upon each local buffer flush.
3. **On-Demand Shard Hydration**: The downstream `SafeTensorsReader` consumes shards lazily. It fetches `manifest.json` on initialization, and resolves requested shards via `hf_hub_download(..., local_dir="cache/activations/...")`, caching files locally on high-speed NVMe scratch storage.
4. **Zero-Knowledge Token Privacy**:
   - Never write tokens into `.py` code, config YAMLs, or committed notebooks.
   - Local: Use OS-level cached tokens from `huggingface-cli login` (`~/.cache/huggingface/token`) or private `.env` files excluded by `.gitignore`.
   - Google Colab: Access encrypted platform secrets via `google.colab.userdata.get('HF_TOKEN')`.
   - Kaggle: Access encrypted platform secrets via `kaggle_secrets.UserSecretsClient().get_secret('HF_TOKEN')`.
   - Principle of Least Privilege: Use fine-grained `read`-only tokens on training/eval workers, reserving `write` tokens exclusively for extraction nodes.
5. **Data Governance**: Maintain repository privacy during the development and benchmarking phases. Transition to public open-science access upon thesis publication.

## Consequences
- Cross-platform portability: Identical training pipelines execute without modification on Local, Kaggle, or Colab.
- Zero startup wait: Training workers can commence mini-batch processing immediately upon downloading the first required shard.
- Fault tolerance: Interrupted cloud sessions lose at most one uncommitted shard buffer (< 500 MB).
- Security guarantee: Tokens are isolated in platform secret vaults and never serialized into source control or shared artifacts.
