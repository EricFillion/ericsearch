from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from torch import device

from ericsearch.utils import es_get_device


class EricVector:
    def __init__(self, model_name_or_path: str, bs: int, d: Optional[device]):
        if d is None:
            d = es_get_device()
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path, device=str(d))
        self.bs = bs

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=self.bs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).to(torch.float16)

    def metadata(self):
        return {"type": "sentence_transformers", "model_name": self.model_name_or_path}
