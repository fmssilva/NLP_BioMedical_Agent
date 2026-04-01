"""
src/embeddings/encoder.py

Sentence encoder using msmarco-distilbert-base-v2 (Lab01 pattern).

Public API:
    Encoder                      -- class: load model once, reuse
    Encoder.encode(texts)        -- batch encode, returns np.ndarray (N, 768)
    Encoder.encode_single(text)  -- single text, returns np.ndarray (768,)

Pattern follows Lab01 exactly:
    AutoTokenizer + AutoModel + mean_pooling (last_hidden_state) + F.normalize
    Does NOT use the sentence-transformers high-level API.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Default constants for local testing
_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v2"


# Mean pool token embeddings weighted by the attention mask.
def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    # model_output.last_hidden_state shape: (batch, seq_len, hidden)
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Encoder:
    """
    Wraps msmarco-distilbert-base-v2 for batch encoding.

    Load once at startup, call .encode() many times.
    Runs on CPU by default (good enough for 4194 docs offline).
    """

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None):
        # pick device: use cuda if available, else cpu
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[encoder] Loading '{model_name}' on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"[encoder] Model loaded. Hidden size: {self.model.config.hidden_size}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embeddings.

        Processes in batches to avoid OOM on large inputs.
        Returns np.ndarray of shape (len(texts), 768).
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input, return_dict=True)

            embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)  # (N, 768)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text, returns shape (768,)."""
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Self-test: python -m src.embeddings.encoder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Step 8 — encoder.py self-test")
    print("=" * 60)

    enc = Encoder()

    # 1. Batch shape check
    sentences = [
        "sleep apnea treatment CPAP",
        "weight loss interventions for obese patients",
        "hepatitis B virus infection antiviral therapy",
    ]
    print(f"\n[1/3] Encoding {len(sentences)} sentences ...")
    vecs = enc.encode(sentences)
    print(f"      Output shape : {vecs.shape}")
    assert vecs.shape == (3, 768), f"Expected (3, 768), got {vecs.shape}"
    print("      Shape check   : OK")

    # 2. L2 norm check — each row must be ~1.0
    norms = np.linalg.norm(vecs, axis=1)
    print(f"      L2 norms      : {norms}")
    assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not 1.0: {norms}"
    print("      L2 norm check : OK")

    # 3. encode_single convenience wrapper
    print("\n[2/3] encode_single check ...")
    single = enc.encode_single(sentences[0])
    assert single.shape == (768,), f"Expected (768,), got {single.shape}"
    assert np.allclose(np.linalg.norm(single), 1.0, atol=1e-5)
    print(f"      encode_single shape: {single.shape}  norm: {np.linalg.norm(single):.6f}  OK")

    # 4. Semantic similarity sanity check — similar sentences should be closer than dissimilar
    print("\n[3/3] Semantic similarity sanity check ...")
    query = enc.encode_single("sleep apnea obstructive treatment")
    relevant = enc.encode_single("CPAP therapy for obstructive sleep apnea patients")
    irrelevant = enc.encode_single("the stock market crashed yesterday after rising interest rates")
    sim_relevant   = float(np.dot(query, relevant))
    sim_irrelevant = float(np.dot(query, irrelevant))
    print(f"      sim(query, relevant)   = {sim_relevant:.4f}")
    print(f"      sim(query, irrelevant) = {sim_irrelevant:.4f}")
    assert sim_relevant > sim_irrelevant, (
        f"Similarity sanity failed: {sim_relevant:.4f} <= {sim_irrelevant:.4f}"
    )
    print("      Similarity order correct: OK")

    print("\n" + "=" * 60)
    print("encoder.py  —  all tests passed")
    print("=" * 60)
