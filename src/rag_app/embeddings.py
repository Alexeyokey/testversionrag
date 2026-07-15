from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        trust_remote_code: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left",
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]

    def _encode_batch(self, texts: Iterable[str]) -> list[list[float]]:
        text_batch = list(texts)
        if not text_batch:
            return []
        batch = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().float().tolist()
