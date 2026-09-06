"""ESM-2 forward hook-based activation extractor."""
from typing import NamedTuple, Iterator, Optional, Sequence
import torch
from transformers import AutoModel, AutoTokenizer

from ptm_sae.config import ModelConfig, ExtractionConfig
from ptm_sae.data.schema import Protein

# Canonical dtype table lookup
_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class ExtractionOutput(NamedTuple):
    uniprot_id: str
    residue_tensor: torch.Tensor        # Shape: (L, hidden_dim) strictly for real amino acids
    mean_pooled_vector: torch.Tensor    # Shape: (hidden_dim,) global context vector


class EsmExtractor:
    """Extracts exact residue-level activations from ESM-2 encoder layers."""

    def __init__(self, model_cfg: ModelConfig, extract_cfg: ExtractionConfig):
        self.model_cfg = model_cfg
        self.extract_cfg = extract_cfg

        # Resolve compute device
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if (model_cfg.device == "auto" and is_cuda) else (model_cfg.device if model_cfg.device != "auto" else "cpu"))

        # Resolve numerical precision dtype
        self.torch_dtype = _DTYPE_MAP.get(model_cfg.dtype, torch.float32)

        # Load tokenizer and frozen PLM backbone
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
        encoder_layers = self.model.encoder.layer

        # Validate layer index bounds
        if target_layer < 0 or target_layer >= len(encoder_layers):
            raise ValueError(f"Target layer {target_layer} out of range (model has {len(encoder_layers)} layers).")

        layer_module = encoder_layers[target_layer]

        def hook_fn(module, input_tensor, output_tensor):
            # Extract raw activations from layer output tuple
            self._captured_activations = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor

        layer_module.register_forward_hook(hook_fn)

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def extract_batch(
        self, records: Sequence[Protein]
    ) -> Iterator[ExtractionOutput]:
        """
        Runs batch through ESM-2 and yields (uniprot_id, residue_tensor, mean_pooled_vector).
        Delimiters (<cls>, <eos>, <pad>) are permanently stripped.
        """
        # 1. Batch tokenize with required start/end special tokens
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

        # 2. Run forward pass without gradient computation
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        activations = self._captured_activations
        if activations is None:
            raise RuntimeError("Forward hook did not capture activations.")

        # 3. Slice real amino acid coordinates (index 1 to seq_len) and compute mean-pooled vector
        for idx, record in enumerate(records):
            seq_len = record.length
            # Token 0 is <cls>; tokens 1..seq_len are real amino acids; token seq_len+1 is <eos>
            residue_acts = activations[idx, 1 : seq_len + 1, :].to("cpu")
            mean_act = residue_acts.mean(dim=0)

            yield ExtractionOutput(
                uniprot_id=record.uniprot_id,
                residue_tensor=residue_acts,
                mean_pooled_vector=mean_act,
            )
