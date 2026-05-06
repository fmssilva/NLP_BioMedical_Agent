"""
Cross-encoder wrapper for sentence-level re-ranking.

Loads a HuggingFace ``AutoModelForSequenceClassification`` checkpoint
(e.g. ``ncbi/MedCPT-Cross-Encoder``) and exposes two scoring methods:
"""

import torch

class RankerModel:
    def __init__(self, name, tokenizer, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def score_query_vs_sentences(self, query, sentences):
        # combine query and sentences into pairs
        pairs = [[query, sent] for sent in sentences]

        with torch.no_grad():
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            scores = self.model(**encoded).logits.squeeze(dim=1).cpu().tolist()

            return sorted(zip(sentences, scores), key=lambda pair: pair[1], reverse=True)