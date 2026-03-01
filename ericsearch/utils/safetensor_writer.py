import dataclasses
import json
import pathlib
import struct
import typing

import safetensors.torch
import torch

# Use a fixed number of header bytes.
# This lets us rewrite the JSON header later
# without needing to shift the data down,
# as we preallocate enough room from the beginning.
NUM_HEADER_BYTES = 1024

INV_TYPES = {v: k for k, v in safetensors.torch._TYPES.items()}


@dataclasses.dataclass
class TensorMetadata:
    partial_shape: typing.Sequence[int]
    dtype: torch.dtype


class SafetensorWriter:
    def __init__(self, path: str | pathlib.Path, meta: TensorMetadata, name: str = "x"):
        self._file = open(path, "wb")
        self._name = name
        self._tensor_metadata = meta
        self._wrote_header = False
        self._len_dim0 = 0
        self._data_byte_count = 0

    def write_header(self):
        if self._wrote_header:
            raise ValueError("already wrote header")

        self._file.write(struct.pack("<Q", NUM_HEADER_BYTES))
        self._file.write(bytes(" " * NUM_HEADER_BYTES, "utf8"))
        self._wrote_header = True

    def write_extend(self, x: torch.Tensor):
        if not self._wrote_header:
            raise ValueError("Must write header before writing data")

        if x.shape[1:] != torch.Size(self._tensor_metadata.partial_shape):
            raise ValueError("Shape mismatch")
        if x.dtype != self._tensor_metadata.dtype:
            raise ValueError("dtype mismatch")

        # Literally just append the data bytes as-is.
        buf = safetensors.torch._tobytes(x, "x")
        self._file.write(buf)
        self._data_byte_count += len(buf)
        self._len_dim0 += len(x)

    def write_append(self, x: torch.Tensor):
        return self.write_extend(torch.unsqueeze(x, 0))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._rewrite_header()
        self._file.close()

    def _rewrite_header(self):
        self._file.seek(8)  # seek past the u64 header byte count.
        self._file.write(
            bytes(
                json.dumps(
                    {
                        self._name: {
                            "dtype": INV_TYPES[self._tensor_metadata.dtype],
                            # Shape is the one field that
                            # DOES change dynamically as we stream writes.
                            # Its the whole reason we have to rewrite the header.
                            "shape": [
                                self._len_dim0,
                                *self._tensor_metadata.partial_shape,
                            ],
                            "data_offsets": [0, self._data_byte_count],
                        }
                    }
                ),
                "utf8",
            )
        )
        # Seek back to end, as this is probably where
        # the caller expects the read pointer to be.
        self._file.seek(0, 2)
