import torch

from ericsearch.train import EricKMeans


class MockRagDatabase:
    """Minimal implementation of the RagDatabase interface
    that works entirely off of one big tensor."""

    def __init__(self, all_vectors: torch.Tensor):
        self.all_vectors = all_vectors

    def __iter__(self):
        for vector in self.all_vectors:
            # Yield batch of 1.
            yield torch.stack([vector])


def test_smoke():
    """Smoke test. Just make sure it doesn't error immediately."""
    kmeans = EricKMeans(n_clusters=1)
    kmeans.train(MockRagDatabase(torch.tensor([[1, 2]], dtype=torch.float32)))
    kmeans.predict(torch.tensor([[1, 2]]))


def test_smoke_excess_clusters():
    """Assert no err when training on less data than clusters"""
    kmeans = EricKMeans(n_clusters=16)
    kmeans.train(MockRagDatabase(torch.tensor([[1, 2], [1, 2]], dtype=torch.float32)))
    kmeans.predict(torch.tensor([[1, 2], [1, 2]]))


def test_simple_clustering():
    """Assert that clustering works! In a very simple case."""
    kmeans = EricKMeans(n_clusters=2)

    kmeans.train(
        MockRagDatabase(
            torch.tensor(
                [
                    [
                        1,
                        0,
                    ],
                    [
                        0,
                        1,
                    ],
                ],
                dtype=torch.float32,
            )
        )
    )

    predicted_labels = kmeans.predict(
        torch.tensor(
            [
                # These should be clustered with the [1,0] training record.
                [0.1, 0.9],
                [0.2, 0.8],
                # This should be clustered with the [0,1] training record.
                [0.9, 0.1],
            ]
        )
    )

    assert predicted_labels[0] == predicted_labels[1]
    assert predicted_labels[2] != predicted_labels[0]
