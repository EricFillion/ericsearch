import contextlib
import math
import pathlib
import typing
from itertools import islice

import safetensors
import torch
from torch.utils.data.dataset import IterableDataset

# Covariant = can return subtypes of T.
T = typing.TypeVar("T", covariant=True)


class SequenceLite(typing.Protocol[T]):
    """Type for an object that implements the important sequence methods,
    without necessarily supporting things like iteration."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int, /) -> T: ...


class SafetensorsSequence:
    """Object accessing a single safetensors file.
    Treats file as a sequence of chunks."""

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        # Larger is faster, but stay within your system's RAM limits.
        chunk_size: int = 10_000,
    ):
        # We should provide a way to eventually close this.
        self._f = safetensors.safe_open(path, framework="pt")
        self._slice = self._f.get_slice(self._f.keys()[0])
        self._chunk_size = chunk_size

    def __len__(self) -> int:
        return math.ceil(self._slice.get_shape()[0] / self._chunk_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._slice[idx * self._chunk_size : (idx + 1) * self._chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._f.__exit__(*exc_info)


def sample_batched_sequence(
    x: SequenceLite[torch.Tensor],
    *,
    output_bs: int,
    n_shuffle_chunks: int = 2,
) -> typing.Iterator[torch.Tensor]:
    """Random sampling over a sequence of (tensor) batches.
    Same effect as shuffle-buffer, but with less bookkeeping."""

    chunk_count = len(x)
    # Shuffle chunks at the macro level.
    chunk_indices = torch.randperm(chunk_count).tolist()

    def batched(iterable, n):
        if n < 1:
            raise ValueError("Empty iterable")
        wrapped_it = iter(iterable)
        while batch := tuple(islice(wrapped_it, n)):
            yield batch

    batched_iter = batched(chunk_indices, n_shuffle_chunks)

    for batch_chunk_indices in batched_iter:
        chunks: list[torch.Tensor] = []
        for batch_chunk_idx in batch_chunk_indices:
            chunk = x[batch_chunk_idx]
            chunks.append(chunk)

        # Concatenate the chunks into a batch
        # and perform a full-item shuffle.
        batch = torch.cat(chunks)
        batch = batch[torch.randperm(len(batch))]

        # Split large output batch into requested size.
        yield from torch.split(batch, output_bs)


class SafetensorsDataset(IterableDataset):
    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        output_bs: int,
        # How large should our file reads from the file itself be?
        # Performance tuning parameter. In theory, smaller values
        # provide slightly purer randomization. But in practice,
        # choosing larger values for performance is probably
        # more important.
        chunk_size: int = 512,
    ) -> None:
        self._seq = SafetensorsSequence(path, chunk_size=chunk_size)
        self._output_bs = output_bs

    def __iter__(self) -> typing.Iterator[torch.Tensor]:
        yield from sample_batched_sequence(
            self._seq, output_bs=self._output_bs
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._seq.__exit__(*exc_info)


@contextlib.contextmanager
def iterate_over_safetensors(path: str | pathlib.Path, internal_chunk_size: int = 1024):
    with SafetensorsSequence(path, chunk_size=internal_chunk_size) as seq:
        yield iterate_over_safetensors_seq(seq)


def iterate_over_safetensors_seq(seq: SafetensorsSequence):
    for i in range(len(seq)):
        batch = seq[i]
        yield from batch
