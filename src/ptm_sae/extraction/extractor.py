"""ESM-2 forward hook-based activation extractor."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Iterator, List, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer

from ptm_sae.config import ModelConfig, ExtractionConfig


@dataclass
class FastaRecord:
    uniprot_id: str
    header: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def parse_fasta(
    fasta_path: str | Path,
    max_sequence_length: int = 1022,
) -> Tuple[List[FastaRecord], List[Tuple[str, int]]]:
    """
    Parse a FASTA file into canonical records.
    Filters sequences exceeding max_sequence_length into a skipped list.
    """
    valid: List[FastaRecord] = []
    skipped: List[Tuple[str, int]] = []

    current_header = None
    current_seq_parts: List[str] = []

    def flush_record():
        if current_header is None:
            return
        seq = "".join(current_seq_parts).strip().upper().replace("*", "")
        # Extract UniProt ID: handles >sp|P04637|... or >P04637 or standard headers
        match = re.search(r">[a-zA-Z0-9_-]+\|([a-zA-Z0-9_-]+)\|", current_header)
        if match:
            u_id = match.group(1)
        else:
            u_id = current_header[1:].split()[0]

        if len(seq) > max_sequence_length:
            skipped.append((u_id, len(seq)))
        elif len(seq) > 0:
            valid.append(FastaRecord(uniprot_id=u_id, header=current_header, sequence=seq))

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                current_header = line
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
        flush_record()

    return valid, skipped


class ExtractionOutput(NamedTuple):
    uniprot_id: str
    residue_tensor: torch.Tensor
    mean_pooled_vector: torch.Tensor


class EsmExtractor:
    """Extracts exact residue-level activations from ESM-2 encoder layers."""

    def __init__(self, model_cfg: ModelConfig, extract_cfg: ExtractionConfig):
        self.model_cfg = model_cfg
        self.extract_cfg = extract_cfg

        # Determine device
        if model_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(model_cfg.device)

        # Determine dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        self.torch_dtype = dtype_map[model_cfg.dtype]

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        self.model = AutoModel.from_pretrained(
            model_cfg.model_name,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._captured_activations: Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self):
        target_layer = self.model_cfg.target_layer
        # HuggingFace ESM has layers under encoder.layer
        encoder_layers = self.model.encoder.layer
        if target_layer < 0 or target_layer >= len(encoder_layers):
            raise ValueError(
                f"Target layer {target_layer} out of range (model has {len(encoder_layers)} layers)."
            )

        layer_module = encoder_layers[target_layer]

        def hook_fn(module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                self._captured_activations = output_tensor[0]
            else:
                self._captured_activations = output_tensor

        layer_module.register_forward_hook(hook_fn)

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def extract_batch(
        self, records: List[FastaRecord]
    ) -> Iterator[ExtractionOutput]:
        """
        Runs batch through ESM-2 and yields (uniprot_id, residue_tensor, cls_tensor).
        Residue tensor strictly has shape (L, hidden_dim) where index i corresponds
        to biological residue i + 1. Delimiters (<cls>, <eos>, <pad>) are stripped.
        """
        sequences = [r.sequence for r in records]
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # self._captured_activations: (batch_size, seq_len_with_special_tokens, hidden_dim)
        activations = self._captured_activations
        if activations is None:
            raise RuntimeError("Forward hook did not capture activations.")

        for idx, record in enumerate(records):
            seq_len = record.length
            # Token 0 is <cls>. Tokens 1 to seq_len are biological amino acids.
            # Token seq_len + 1 is <eos>. Remaining tokens are <pad>.
            residue_acts = activations[idx, 1 : seq_len + 1, :].to("cpu")
            mean_act = residue_acts.mean(dim=0)

            yield ExtractionOutput(
                uniprot_id=record.uniprot_id,
                residue_tensor=residue_acts,
                mean_pooled_vector=mean_act,
            )
