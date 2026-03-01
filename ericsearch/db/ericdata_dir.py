from dataclasses import dataclass
import json
import pathlib
import typing

import safetensors.torch
import torch
from torch import device

from ericsearch.utils.eric_vector import EricVector
from ericsearch.utils.types import EmbeddingsModel


@dataclass(kw_only=True)
class Cluster:
    embeddings: torch.Tensor
    texts: list[str]
    metas: list[typing.Any]


@dataclass(kw_only=True)
class Centroids:
    ivf0_centroids: torch.Tensor

    ivf1_centroids: list[torch.Tensor]
    """ ivf1_centroids[i] is a tensor of centroids
        for ivf0_c==i. """


class EricSearchDirStateReader:
    """Loads EricSearch state from a directory.
    The canonical dataset itself is only loaded on demand,
    as it grows quite large. But our ensemble of lightweight
    clustering models can all be loaded eagerly."""

    def __init__(self, root: str | pathlib.Path, d: device) -> None:
        self.root = pathlib.Path(root)
        self.device = d

    def load_centroids(self) -> Centroids:
        all_centroids = safetensors.torch.load_file(
            filename=self.root / "centroids.safetensors", device=str(self.device)
        )
        return Centroids(
            ivf0_centroids=all_centroids["ivf0"],
            ivf1_centroids=list(all_centroids["ivf1"]),
        )

    def load_cluster(self, ivf0_c: int, ivf1_c: int) -> Cluster:
        """Load a specific leaf cluster into memory, including all embeddings and texts."""
        cluster_dir = self.root / "ivf" / str(ivf0_c) / str(ivf1_c)

        embeddings = safetensors.torch.load_file(
            filename=cluster_dir / "embeddings.safetensors", device=str(self.device)
        )["embeddings"]
        with open(cluster_dir / "data.jsonl") as f:
            records = [json.loads(line) for line in f if line.strip()]

        texts = [record["text"] for record in records]
        metas = [record.get("metadata") for record in records]

        return Cluster(embeddings=embeddings, texts=texts, metas=metas)

    def load_embeddings_model(self) -> EmbeddingsModel:
        with open(self.root / "eric_details.json") as f:
            config = json.load(f)["embeddings_model"]

        if config["type"] != "sentence_transformers":
            raise ValueError("expected type = sentence_transformers")

        return EricVector(config["model_name"], bs=16, d=self.device)
