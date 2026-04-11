import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

######################################################################
## Sentence/batch encoder — pluggable pooling + L2 normalisation.
## Supports three pooling modes:
##   "mean"            — mean over ALL tokens (CLS, content, SEP; no PAD)
##   "mean_no_special" — mean over content tokens only (CLS and SEP excluded)
##   "cls"             — [CLS] token only (position 0)
##
## Singleton cache per (model_name, device, pooling_mode) triple.
######################################################################

# Suppress the harmless "UNEXPECTED key: embeddings.position_ids" warning that
# MPNet checkpoints emit when loaded via AutoModel.  MPNet saves position_ids as
# a persistent buffer in the checkpoint; the generic AutoModel loader flags it as
# unexpected, but the buffer is recomputed at runtime and the warning is safe to
# ignore.  We only silence the specific transformers loading logger to avoid
# hiding real errors from other modules.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Default constants for local testing
_DEFAULT_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v2"

# Valid pooling modes
POOLING_MEAN            = "mean"
POOLING_MEAN_NO_SPECIAL = "mean_no_special"
POOLING_CLS             = "cls"
VALID_POOLING_MODES     = {POOLING_MEAN, POOLING_MEAN_NO_SPECIAL, POOLING_CLS}


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over ALL real tokens (CLS + content + SEP), excluding PAD."""
    token_embeddings = model_output.last_hidden_state       # (B, L, H)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _mean_pooling_no_special(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool over content tokens only — CLS (position 0) and SEP (last real
    token per sequence) are excluded.  Falls back to CLS for degenerate
    sequences that contain only CLS+SEP (length==2).
    """
    token_embeddings = model_output.last_hidden_state       # (B, L, H)
    mask = attention_mask.clone().float()                   # (B, L) — don't mutate original

    # zero CLS at position 0
    mask[:, 0] = 0.0

    # zero SEP: last position where original attention_mask==1
    last_real = (attention_mask.sum(dim=1) - 1).clamp(min=0)   # (B,)
    batch_idx = torch.arange(mask.size(0), device=mask.device)
    mask[batch_idx, last_real] = 0.0

    # fallback: if no content tokens remain (seq_len==2), restore CLS
    row_sums = mask.sum(dim=1, keepdim=True)                # (B, 1)
    fallback  = (row_sums == 0).squeeze(1)                  # (B,) bool
    if fallback.any():
        mask[fallback, 0] = 1.0
        row_sums = mask.sum(dim=1, keepdim=True)

    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(row_sums, min=1e-9)


def _cls_pooling(model_output) -> torch.Tensor:
    """[CLS] token only (MedCPT official approach: last_hidden_state[:, 0, :])."""
    return model_output.last_hidden_state[:, 0, :]          # (B, H)


# Module-level singleton cache — keyed by (model_name, device, pooling_mode).
_encoder_cache: dict[tuple[str, str, str], "Encoder"] = {}


class Encoder:
    """
    Singleton transformer encoder (one instance per model_name + device + pooling_mode).
    Call ``Encoder(model_name)`` repeatedly — returns the already-loaded instance.

    pooling_mode options (use the POOLING_* constants):
        "mean"            (default) — mean over all real tokens incl. CLS/SEP
        "mean_no_special" — mean over content tokens only (CLS and SEP excluded)
        "cls"             — CLS token only (MedCPT official approach)
    """

    def __new__(
        cls,
        model: str | tuple[str, str, int] = _DEFAULT_MODEL_NAME,
        device: str | None = None,
        pooling_mode: str = POOLING_MEAN,
    ):
        if pooling_mode not in VALID_POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {VALID_POOLING_MODES}, got '{pooling_mode}'")
        model_name = model[1] if isinstance(model, tuple) else model
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (model_name, resolved_device, pooling_mode)
        if key in _encoder_cache:
            return _encoder_cache[key]
        instance = object.__new__(cls)
        instance._initialised = False
        _encoder_cache[key] = instance
        return instance

    def __init__(
        self,
        model: str | tuple[str, str, int] = _DEFAULT_MODEL_NAME,
        device: str | None = None,
        pooling_mode: str = POOLING_MEAN,
    ):
        if self._initialised:
            return                                 # already loaded — skip entirely
        model_name = model[1] if isinstance(model, tuple) else model
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device       = resolved_device
        self.model_name   = model_name
        self.pooling_mode = pooling_mode

        print(f"[encoder] Loading '{model_name}' on {resolved_device} (pooling={pooling_mode}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self._initialised = True
        print(f"[encoder] Model loaded. Hidden size: {self.model.config.hidden_size}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embeddings.
        Processes in batches to avoid OOM on large inputs.
        Returns np.ndarray of shape (len(texts), hidden_size).

        max_length is capped at the model's own positional embedding size so
        models with a 512-token limit (e.g. MedCPT/BERT) don't overflow.
        """
        max_len = getattr(self.tokenizer, "model_max_length", None) or \
                  getattr(self.model.config, "max_position_embeddings", 512)
        # Some tokenizers report 1e30 (no real limit); clamp to a safe ceiling.
        if max_len > 10_000:
            max_len = 512

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input, return_dict=True)

            # dispatch to the selected pooling strategy
            if self.pooling_mode == POOLING_CLS:
                embeddings = _cls_pooling(model_output)
            elif self.pooling_mode == POOLING_MEAN_NO_SPECIAL:
                embeddings = _mean_pooling_no_special(model_output, encoded_input["attention_mask"])
            else:  # POOLING_MEAN (default)
                embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)  # (N, hidden_size)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text, returns shape (hidden_size,)."""
        return self.encode([text])[0]




