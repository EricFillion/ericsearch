import typing
from dataclasses import dataclass

import torch

# For our purposes, a dataloader is just a tensor iterable.
# Note that each iterator from the iterator shall have its own state.
DataLoaderLike = typing.Iterable[torch.Tensor]


class EmbeddingsModel(typing.Protocol):
    def encode(self, texts: list[str]) -> torch.Tensor: ...

    def metadata(self) -> dict:
        """Get metadata for this embeddings model.
        Typically saved to disk for reference upon future load.
        """

@dataclass(kw_only=True)
class SearchResult:
    """Object representing a block-level result
    of text found in a larger document.

    The top-scoring sentence is also included,
    in case the caller is interested in that specifically."""

    text: str
    score: float
    best_sentence: str
    metadata: dict

@dataclass(kw_only=True)
class RankerResult:
    text: str
    score: float
    best_sentence: str
    metadata: dict


@dataclass(kw_only=True)
class EricDocument:
    """Object representing a block-level result
    of text found in a larger document.

    The top-scoring sentence is also included,
    in case the caller is interested in that specifically."""

    text: str
    score: float = 0.0
    metadata: dict = 0
