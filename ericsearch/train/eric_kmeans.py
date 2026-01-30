# Key to understanding this code:
# shape[0] is number of points
# shape[1] is number of dimensions to each point

import os
from typing import Optional

import safetensors.torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ericsearch.utils import EricTimer, take_from_dataloader, DataLoaderLike


class EricKMeans:
    def __init__(
        self,
        n_clusters: int = 16,
        n_features: int = 384,
        max_iter: int = 16,
        tolerance: float = 1e-2,
        seed: Optional[int] = None,
        eric_timer: Optional[EricTimer] = None,
        size_penalty: float = 0.1,
    ):
        max_iter = max_iter
        self.n_clusters = int(n_clusters)
        self.n_features = int(n_features)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.seed = seed
        self.centroids: Optional[torch.Tensor] = None  # [k, D], float32

        self.eric_timer = (
            eric_timer
            if eric_timer is not None
            else EricTimer("eric_search/kmeans_timer.csv", False)
        )
        self.size_penalty = size_penalty

    @torch.no_grad()
    def train(
        self,
        dataloader: DataLoader[torch.Tensor],
        # data: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        show_progress: bool = False,
    ) -> None:
        dev = device

        with self.eric_timer.section("eric_k_means", "initialize"):
            centroids = init_random_centroids(
                dataloader=dataloader,
                n_clusters=self.n_clusters,
                n_features=self.n_features,
            )

        # Upload data to GPU in float32 form.
        # x = data.to(dev, dtype=dtype, non_blocking=True)
        # Shape is [k,D]
        with self.eric_timer.section("eric_k_means", "moving centroids to device"):
            centroids = centroids.to(dev, dtype=dtype, non_blocking=True)

        # Normalize input data along dimension 1.
        # This ensures that each feature is normalized across the batch.
        with self.eric_timer.section("eric_k_means", "normalize_init"):
            centroids = F.normalize(centroids, dim=1, eps=1e-8)

        # Bootstrap prev_centroids
        with self.eric_timer.section("eric_k_means", "alloc_buffers"):
            prev_centroids = centroids.clone()

            # Preallocate reusable buffers.
            # These get new values each loop iteration.

            # shape is [k, D]
            sums = torch.empty_like(centroids)

        # Numerical method iterations
        for _ in tqdm(
            range(self.max_iter),
            disable=not show_progress,
            desc="KMeans classification iteration: ",
        ):
            new_centroids, counts = iterate_centroids(
                dataloader=dataloader,
                centroids=centroids,
                prev_centroids=prev_centroids,
                sums=sums,
                n_clusters=self.n_clusters,
                eric_timer=self.eric_timer,
                size_penalty=self.size_penalty,
            )

            # Normalize new centroids, as we need scales to match
            # before we can compute distance
            with self.eric_timer.section("eric_k_means", "normalize new centroids"):
                new_centroids = F.normalize(new_centroids, dim=1, eps=1e-8)

            with self.eric_timer.section("eric_k_means", "reinitialize"):
                # Re-seed all clusters with zero assignments,
                # by sampling points from the largest cluster.
                if (counts == 0).any():
                    empty_cluster_indices = (counts == 0).nonzero(as_tuple=True)[0]

                    try:
                        sampled = sample_from_cluster(
                            dataloader=dataloader,
                            cluster_idx=counts.argmax(),
                            limit=len(empty_cluster_indices),
                            dev=dev,
                            dtype=dtype,
                            centroids=new_centroids,
                        )
                        centroids[empty_cluster_indices[: len(sampled)]] = sampled

                    except Exception:
                        # if it fails just use a simple random selection
                        fallback_list = take_from_dataloader(
                            dataloader, limit=len(empty_cluster_indices)
                        )
                        if len(fallback_list) > 0:
                            sampled = torch.stack(fallback_list).to(
                                device=dev, dtype=dtype, non_blocking=True
                            )
                            sampled = F.normalize(sampled, dim=1, eps=1e-8)
                            centroids[empty_cluster_indices[: len(sampled)]] = sampled

            # Commit iteration result.
            centroids = new_centroids
            prev_centroids = new_centroids

        self.centroids = centroids
        self.eric_timer.report()

    @torch.no_grad()
    def predict_multi(self, data: torch.Tensor, *, k: int) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("Train the KMeans first by calling train()")

        x = data.to(
            device=self.centroids.device, dtype=self.centroids.dtype, non_blocking=True
        )
        # Normalize input to match normalized centroids
        x = F.normalize(x, dim=1, eps=1e-8)
        # To find the corresponding centroid,
        # we simply perform a matrix multiplication.
        sims = x @ self.centroids.T

        return sims.argsort(dim=1, descending=True)[:, :k]

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        return self.predict_multi(data, k=1)[:, 0]

    def save(self, path: str | os.PathLike[str]) -> None:
        if self.centroids is None:
            raise ValueError("EricKMeans.centroids is None")

        safetensors.torch.save_file(
            tensors={"centroids": self.centroids}, filename=path
        )

    def load(self, path: str | os.PathLike[str]) -> None:
        self.centroids = safetensors.torch.load_file(path)["centroids"]


def init_random_centroids(
    dataloader: DataLoader[torch.Tensor],
    n_clusters: int,
    n_features: int,
) -> torch.Tensor:
    initial_centroids: list[torch.Tensor] = []
    # Try to sample without replacement, but there are edge cases
    # where we end up duplicating samples due to N<n_clusters.
    # We support these for correctness/completeness more than anything.
    while len(initial_centroids) < n_clusters:
        # Assume that dataloader is shuffled.
        batch = take_from_dataloader(
            dataloader, limit=(n_clusters - len(initial_centroids))
        )
        if len(batch) == 0:
            return torch.zeros(size=(n_clusters, n_features), dtype=torch.float32)

        # print(f'init_random_centroids(): {batch[0].shape=}')
        initial_centroids.extend(batch)

    return torch.stack(initial_centroids)


def iterate_centroids(
    *,
    dataloader: DataLoaderLike,
    centroids: torch.Tensor,
    prev_centroids: torch.Tensor,
    sums: torch.Tensor,
    n_clusters: int,
    eric_timer: EricTimer,
    size_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the dataset against itself.

    Then, the value of each cluster becomes
    the mean of all the points that it captured.
    """

    with eric_timer.section("eric_k_means", "iterate_centroids init"):
        device = centroids.device
        dtype = centroids.dtype
        # Recall shapes are [k] for both.
        # Zero them out before we start accumulating.
        sums.zero_()
        counts = torch.zeros(n_clusters, dtype=torch.int64, device=device)
    # Accumulate sums and counts in a loop.
    # Such that we can multiply smaller values
    # instead of one giant value.

    data_iter = iter(dataloader)

    while True:
        with eric_timer.section("file_operations", "EricKMeans: next_batch"):
            try:
                x = next(data_iter)
            except StopIteration:
                break

        with eric_timer.section("eric_k_means", "unpack"):
            if isinstance(x, (list, tuple)):
                x = x[0]
        with eric_timer.section("eric_k_means", "x to_device"):
            x = x.to(device=device, dtype=dtype, non_blocking=True)
        # Normalize input data along dimension 1.
        with eric_timer.section("eric_k_means", "x normalize"):
            x = F.normalize(x, dim=1, eps=1e-8)
        # Find predicted labels from centroids (matrix multiplication).
        with eric_timer.section("eric_k_means", "matrix multiplication"):
            sims = x @ centroids.T

        if size_penalty > 0.0:
            with eric_timer.section("eric_k_means", "size_penalty"):
                counts_f = counts.to(dtype=sims.dtype)
                max_count = counts_f.max().clamp(min=1.0)
                penalty = size_penalty * (counts_f / max_count)
                sims = sims - penalty

        # Shape is [num_clusters]. Each label is a cluster idx.
        with eric_timer.section("eric_k_means", "argmax"):
            labels = sims.argmax(dim=1)

        # Essentially adds features based on label.
        # sums[ labels[i] ] += x[i]
        # sums.scatter_add_(0, labels[:, None].expand(-1, n_clusters), x)

        # for idx,label in enumerate(labels):
        #     sums[label] += x[idx]
        with eric_timer.section("eric_k_means", "index_add"):
            sums.index_add_(0, labels, x)

        # Count how many labels are
        with eric_timer.section("eric_k_means", "bincount"):
            counts += torch.bincount(labels, minlength=n_clusters)

    with eric_timer.section("eric_k_means", "iterate_centroids final"):
        # Clamp counts to [0,1], essentially.
        nonempty = counts > 0
        counts_clamped = counts.clone()
        # Fill any 0s with 1s, to avoid division by 0 in the sum/count line.
        counts_clamped.masked_fill_(~nonempty, 1.0)

        # Take the new centroids where count is nonempty.
        new_centroids = sums / counts_clamped.unsqueeze(1)
        new_centroids = torch.where(
            nonempty.unsqueeze(1), new_centroids, prev_centroids
        )

    return new_centroids, counts


def sample_from_cluster(
    *,
    dataloader: DataLoader[torch.Tensor],
    cluster_idx: int | torch.Tensor,
    limit: int,
    dev: torch.device | None,
    dtype: torch.dtype,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """From a batches-of-points dataloader,
    find up to `limit` points that fall into `cluster_idx`."""

    batches_in_cluster: list[torch.Tensor] = []
    num_points = 0

    # Iterate over batches from dataloader.
    for x in dataloader:
        # Run inference on batch.
        x = x.to(device=dev, dtype=dtype, non_blocking=True)
        x_norm = F.normalize(x, dim=1, eps=1e-8)

        sims = x_norm @ centroids.T
        labels = sims.argmax(dim=1)

        # Add points belonging to this cluster, if any.
        mask = labels == cluster_idx
        if mask.any():
            batches_in_cluster.append(x_norm[mask])

        num_points += int(mask.sum().item())
        if num_points >= limit:
            break

    # Wrap up result in a one 2D tensor of shape [N,D].
    points_in_cluster = torch.concat(batches_in_cluster)[:limit]

    # Normalize feature within each point against other features in same point.
    points_in_cluster = F.normalize(points_in_cluster, dim=1, eps=1e-8)

    return points_in_cluster
