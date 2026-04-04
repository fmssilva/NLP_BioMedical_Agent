import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


######################################################################
## Sentence/batch encoder — mean pooling + L2 normalisation.
## Singleton cache per (model_name, device) pair to avoid re-loading.
######################################################################

# Default constants for local testing
_DEFAULT_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v2"


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool token embeddings weighted by the attention mask.
    """
    # model_output.last_hidden_state shape: (batch, seq_len, hidden)
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Module-level singleton cache — keyed by (model_name, device).
_encoder_cache: dict[tuple[str, str], "Encoder"] = {}


class Encoder:
    """
    Singleton transformer encoder (one instance per model_name + device).
    Call ``Encoder(model_name)`` repeatedly — returns the already-loaded
    instance on subsequent calls, avoiding redundant downloads and GPU allocations.
    Runs on CPU by default; use ``device="cuda"`` to move to GPU.
    """

    def __new__(cls, model: str | tuple[str, str, int] = _DEFAULT_MODEL_NAME, device: str | None = None):
        model_name = model[1] if isinstance(model, tuple) else model
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (model_name, resolved_device)
        if key in _encoder_cache:
            return _encoder_cache[key]
        instance = object.__new__(cls)
        instance._initialised = False
        _encoder_cache[key] = instance
        return instance

    def __init__(self, model: str | tuple[str, str, int] = _DEFAULT_MODEL_NAME, device: str | None = None):
        if self._initialised:
            return                                 # already loaded — skip entirely
        model_name = model[1] if isinstance(model, tuple) else model
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model_name = model_name

        print(f"[encoder] Loading '{model_name}' on {resolved_device} ...")
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
                return_tensors="pt", # this tells the tokenizer to return PyTorch tensors instead of lists or numpy arrays
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input, return_dict=True)

            embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)  # (N, hidden_size)


    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text, returns shape (hidden_size,)."""
        return self.encode([text])[0]



