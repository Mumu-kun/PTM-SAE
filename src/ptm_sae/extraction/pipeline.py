"""End-to-end activation extraction pipeline runner with multi-GPU parallelization, cloud resumption and OOM fallback."""

import argparse
import queue
import threading
from collections.abc import Sequence
from pathlib import Path

import torch
from tqdm import tqdm

from ptm_sae.config import PipelineConfig
from ptm_sae.data import parse_uniprot_fasta
from ptm_sae.data.fetcher import fetch_uniprot_human_proteome
from ptm_sae.data.schema import Protein
from ptm_sae.extraction.extractor import EsmExtractor
from ptm_sae.extraction.sharder import SafeTensorsSharder


def run_extraction_pipeline(
    config: PipelineConfig,
    fasta_path: str | Path | None = None,
    proteins: Sequence[Protein] | None = None,
    max_proteins: int | None = None,
) -> dict:
    """
    Executes end-to-end ESM-2 activation extraction:
    1. Loads candidate FASTA proteins.
    2. Initializes SafeTensors sharder and skips already committed proteins.
    3. Resolves compute devices (single-GPU or multi-GPU pool across all visible GPUs).
    4. Runs batched PLM extraction with automated CUDA OOM fallback.
    5. Commits remaining buffers and syncs manifest to local cache and remote hub asynchronously.
    """
    # 1. Resolve candidate protein sequences
    if proteins is None:
        if fasta_path is None:
            fasta_path = fetch_uniprot_human_proteome(out_dir="data/raw")

        target_fasta = Path(fasta_path)
        if not target_fasta.exists():
            raise FileNotFoundError(f"FASTA file not found at {target_fasta}")

        valid_proteins, skipped = parse_uniprot_fasta(
            target_fasta,
            max_sequence_length=config.extraction.max_sequence_length,
        )
        print(
            f"[Pipeline] Parsed {len(valid_proteins)} valid proteins (skipped {len(skipped)} exceeding length limit)."
        )
    else:
        valid_proteins = list(proteins)

    if max_proteins is not None:
        valid_proteins = valid_proteins[:max_proteins]

    # 2. Initialize SafeTensors Sharded Buffer Manager
    print(
        f"[Pipeline] Initializing SafeTensors Sharder (output: {config.sharding.output_dir})..."
    )
    sharder = SafeTensorsSharder(config.sharding)

    # 3. Filter out proteins already committed in existing manifest
    uncommitted = [p for p in valid_proteins if not sharder.is_committed(p.uniprot_id)]
    print(
        f"[Pipeline] Total proteins: {len(valid_proteins)} | Already committed: {len(valid_proteins) - len(uncommitted)} | Remaining: {len(uncommitted)}"
    )

    if not uncommitted:
        print("[Pipeline] All proteins are already committed. Extraction is complete.")
        return sharder.manifest

    # 4. Resolve compute devices for single-GPU or multi-GPU execution
    if config.model.device == "cpu":
        devices = [torch.device("cpu")]
    elif config.model.device.startswith("cuda:") or (
        config.model.device == "cuda" and torch.cuda.device_count() <= 1
    ):
        devices = [
            torch.device(
                config.model.device if config.model.device != "auto" else "cuda:0"
            )
        ]
    elif torch.cuda.is_available():
        available = torch.cuda.device_count()
        requested = config.extraction.num_gpus or available
        count = max(1, min(available, requested))
        devices = [torch.device(f"cuda:{i}") for i in range(count)]
    else:
        devices = [torch.device("cpu")]

    batch_size = config.extraction.batch_size
    pbar = tqdm(total=len(uncommitted), desc="Extracting activations", unit="protein")

    # 5a. Single-Device Execution Path
    if len(devices) == 1:
        single_dev = devices[0]
        print(f"[Pipeline] Single-device mode: {single_dev} (batch_size={batch_size})")
        dev_cfg = config.model.model_copy()
        dev_cfg.device = str(single_dev)
        extractor = EsmExtractor(dev_cfg, config.extraction)

        for i in range(0, len(uncommitted), batch_size):
            batch = uncommitted[i : i + batch_size]
            try:
                for out in extractor.extract_batch(batch):
                    sharder.add_protein(
                        out.uniprot_id, out.residue_tensor, out.mean_pooled_vector
                    )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"\n[Warning] CUDA OOM on batch {i}. Recovering with batch_size=1..."
                )
                torch.cuda.empty_cache()
                for single_protein in batch:
                    for out in extractor.extract_batch([single_protein]):
                        sharder.add_protein(
                            out.uniprot_id, out.residue_tensor, out.mean_pooled_vector
                        )

            pbar.update(len(batch))

    # 5b. Multi-GPU Parallel Execution Path
    else:
        print(
            f"[Pipeline] Multi-GPU mode: parallelizing across {len(devices)} GPUs ({', '.join(str(d) for d in devices)})"
        )
        extractors: dict[str, EsmExtractor] = {}
        for dev in devices:
            dev_cfg = config.model.model_copy()
            dev_cfg.device = str(dev)
            extractors[str(dev)] = EsmExtractor(dev_cfg, config.extraction)

        batch_queue: queue.Queue = queue.Queue()
        for i in range(0, len(uncommitted), batch_size):
            batch_queue.put(uncommitted[i : i + batch_size])

        pbar_lock = threading.Lock()

        def _worker(dev_str: str):
            inst = extractors[dev_str]
            while True:
                try:
                    batch = batch_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    for out in inst.extract_batch(batch):
                        sharder.add_protein(
                            out.uniprot_id, out.residue_tensor, out.mean_pooled_vector
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    for single_protein in batch:
                        for out in inst.extract_batch([single_protein]):
                            sharder.add_protein(
                                out.uniprot_id,
                                out.residue_tensor,
                                out.mean_pooled_vector,
                            )

                with pbar_lock:
                    pbar.update(len(batch))
                batch_queue.task_done()

        threads = [
            threading.Thread(target=_worker, args=(str(d),), name=f"gpu_worker_{d}")
            for d in devices
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    pbar.close()

    # 6. Flush remaining buffers and ensure background uploads finish
    sharder.close()
    print(
        f"[Pipeline] Extraction finalized. Total tokens in manifest: {sharder.manifest['total_tokens']:,}"
    )
    return sharder.manifest


def main():
    parser = argparse.ArgumentParser(
        description="Run ESM-2 Activation Extraction Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev_8m.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Path to FASTA file (defaults to reviewed human proteome)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to extract (for testing)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="Number of GPUs to utilize"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device ('cuda', 'cpu', 'auto')",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.device:
        cfg.model.device = args.device
    if args.num_gpus:
        cfg.extraction.num_gpus = args.num_gpus

    run_extraction_pipeline(
        config=cfg,
        fasta_path=args.fasta,
        max_proteins=args.max_proteins,
    )


if __name__ == "__main__":
    main()
