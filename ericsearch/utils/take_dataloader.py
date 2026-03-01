import typing

from torch.utils.data import DataLoader

T = typing.TypeVar("T")


def take_from_dataloader(loader: DataLoader[T], limit: int) -> list[T]:
    """Take upto `limit` items from `loader`,
    regardless of the loader's batch size.."""

    taken: list[T] = []
    for batch in loader:
        for item in batch:
            taken.append(item)
            if len(taken) >= limit:
                return taken

    return taken
