import pathlib

import safetensors.torch
import torch

from ericsearch.utils.safetensor_writer import (
    SafetensorWriter,
    TensorMetadata,
)
from ericsearch.train.safetensors_dataset import (
    SafetensorsDataset,
    SafetensorsSequence,
)


def test_basic(tmp_path: pathlib.Path):
    with SafetensorWriter(
        tmp_path / "x.safetensors",
        TensorMetadata(partial_shape=[2], dtype=torch.float32),
    ) as writer:
        writer.write_header()
        writer.write_extend(torch.tensor([[1.0, 3.0]], dtype=torch.float32))
        writer.write_extend(torch.tensor([[2.0, 4.0]], dtype=torch.float32))

    x = safetensors.torch.load_file(tmp_path / "x.safetensors")["x"]

    assert torch.equal(x, torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32))


def test_seq(tmp_path: pathlib.Path):
    safetensors.torch.save_file(
        {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}, tmp_path / "x.safetensors"
    )

    seq = SafetensorsSequence(tmp_path / "x.safetensors", chunk_size=2)
    assert len(seq) == 2
    assert seq[0].tolist() == [1.0, 2.0]
    assert seq[1].tolist() == [3.0, 4.0]


def test_ds(tmp_path: pathlib.Path):
    safetensors.torch.save_file(
        {"x": torch.tensor(list(range(1000)))}, tmp_path / "x.safetensors"
    )

    seq = SafetensorsDataset(tmp_path / "x.safetensors", output_bs=2)
    all_batches = list(seq)
    all_items = [item for batch in all_batches for item in batch.tolist()]
    # Assert that some kind of shuffling happened.
    assert all_items != list(range(1000))
    # Assert that all expected items are present.
    assert sorted(all_items) == list(range(1000))
