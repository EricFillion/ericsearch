import contextlib
import dataclasses
import json
import logging
import math
import pathlib
import typing
from itertools import islice

import safetensors.torch
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm.auto import tqdm

from ericsearch.train.eric_kmeans import EricKMeans
from ericsearch.utils import  SafetensorWriter, TensorMetadata, EricTimer
from ericsearch.train.safetensors_dataset import (
    SafetensorsDataset,
    iterate_over_safetensors,
)
from ericsearch.utils import EmbeddingsModel

EmbedFn = typing.Callable[[list[str]], torch.Tensor]


class EricDataTrainArgs(typing.TypedDict):
    k0: int
    k1: int
    input_count: int
    bs: int
    bs_kmeans: int
    input_dir: pathlib.Path
    out_dir: pathlib.Path
    embeddings_model: EmbeddingsModel
    device: torch.device
    logger: logging.Logger
    eric_timer: EricTimer


def train_eric_search(**kwargs):
    eric_timer = kwargs["eric_timer"]
    logger = kwargs["logger"]
    n_features = kwargs["embeddings_model"].encode(["x"]).shape[1]
    dtype = kwargs["embeddings_model"].encode(["x"]).dtype

    # Save embeddings for each input.
    embed_inputs(
        input_dir=kwargs["input_dir"],
        out_dir=kwargs["out_dir"] / "artifacts" / "embeddings",
        embed_fn=kwargs["embeddings_model"].encode,
        input_count=kwargs["input_count"],
        bs=kwargs["bs"],
        eric_timer=eric_timer,
    )

    eric_timer.report()

    with eric_timer.section("eric_k_means", "init ivf0 clusters"):
        # Train top level cluster.
        ivf0_cluster = EricKMeans(n_clusters=kwargs["k0"], eric_timer=eric_timer)
        logger.info("\nTraining top level cluster:")

    with torch_dataloader_for_dir(
        kwargs["out_dir"] / "artifacts" / "embeddings",
        bs=kwargs["bs_kmeans"],
    ) as ivf_0_data:
        ivf0_cluster.train(ivf_0_data, device=kwargs["device"], show_progress=True)

    logger.info("\n Splitting top level cluster:")
    with eric_timer.section("file_operations", "Generic: split_dataset: ivf0"):
        split_dataset(
            input_dir=kwargs["out_dir"] / "artifacts" / "embeddings",
            out_dir=lambda i: kwargs["out_dir"] / "artifacts" / "c0" / str(i),
            cluster=ivf0_cluster,
            bs=kwargs["bs"],
            n_features=n_features,
            dtype=dtype,
        )
    eric_timer.report()

    logger.info("\nTraining secondary clusters:")

    with eric_timer.section("eric_k_means", "init ivf1 clusters"):
        ivf1_clusters = [
            EricKMeans(n_clusters=kwargs["k1"], eric_timer=eric_timer)
            for _ in range(kwargs["k0"])
        ]

    for c0, ivf1_cluster in tqdm(
        enumerate(ivf1_clusters), total=len(ivf1_clusters), desc="K1 Clustering"
    ):
        with torch_dataloader_for_dir(
            kwargs["out_dir"] / "artifacts" / "c0" / str(c0),
            bs=kwargs["bs_kmeans"],
        ) as ivf1_dataloader:
            ivf1_cluster.train(
                ivf1_dataloader,
                device=kwargs["device"],
            )

    eric_timer.report()

    logger.info("\nSaving: ")
    with eric_timer.section("file_operations", "Generic: ivf1 save"):
        # Save centroids, now that they're all trained.
        safetensors.torch.save_file(
            {
                "ivf0": required(ivf0_cluster.centroids),
                "ivf1": torch.stack(
                    [required(ivf1_cluster.centroids) for ivf1_cluster in ivf1_clusters]
                ),
            },
            pathlib.Path(kwargs["out_dir"]) / "centroids.safetensors",
        )

    eric_timer.report()

    with eric_timer.section("file_operations", "Generic: split_dataset: ivf1"):
        # Further split dataset according to secondary cluster.
        for c0 in range(kwargs["k0"]):
            split_dataset(
                input_dir=kwargs["out_dir"] / "artifacts" / "c0" / str(c0),
                out_dir=lambda c1: (
                    kwargs["out_dir"] / "ivf" / str(c0) / str(c1)
                ),
                cluster=ivf1_clusters[c0],
                bs=kwargs["bs"],
                n_features=n_features,
                dtype=dtype,
            )
    eric_timer.report()

    # Infer the number features the embeddings model uses
    # by embedding the sample text 'x' and checking the shape.
    n_features = kwargs["embeddings_model"].encode(["x"]).shape[-1]

    with eric_timer.section("file_operations", "Generic: write final config file"):
        # Write configuration such that the `EricData.load()` caller
        # doesn't need to remember what their config was.
        with open(kwargs["out_dir"] / "eric_details.json", "w") as f:
            json.dump(
                {
                    "k0": kwargs["k0"],
                    "k1": kwargs["k1"],
                    # "version": version("ericsearch"), #todo uncomment
                    "embeddings_model": kwargs["embeddings_model"].metadata(),
                },
                f,
                indent=4,
            )
    eric_timer.report()


@dataclasses.dataclass
class Batch:
    texts: list[str]
    embeddings: torch.Tensor
    metas: list[typing.Any]


def mini_dataloader_for_dir(dir: pathlib.Path) -> typing.Iterator[Batch]:
    # Variant for batch size 1.
    with iterate_over_safetensors(dir / "embeddings.safetensors") as embeddings_it:
        with open(dir / "data.jsonl") as f:
            while line := f.readline():
                if line.strip():
                    record = json.loads(line)
                    embedding = next(embeddings_it)
                    yield Batch(
                        texts=[record["text"]],
                        embeddings=embedding,
                        metas=[record.get("metadata")],
                    )


def dataloader_for_dir(dir: pathlib.Path, bs: int) -> typing.Iterable[Batch]:
    buf: list[Batch] = []

    def make_batch() -> Batch:
        batch = Batch(
            texts=[b.texts[0] for b in buf],
            embeddings=torch.stack([b.embeddings for b in buf]),
            metas=[b.metas[0] for b in buf],
        )
        buf.clear()
        return batch

    for minibatch in mini_dataloader_for_dir(dir):
        buf.append(minibatch)
        if len(buf) >= bs:
            yield make_batch()

    if buf:
        yield make_batch()


@contextlib.contextmanager
def torch_dataloader_for_dir(
    dir: pathlib.Path, bs: int
) -> typing.Iterator[TorchDataLoader]:
    with SafetensorsDataset(
        dir / "embeddings.safetensors", output_bs=bs
    ) as dataset:
        yield TorchDataLoader(dataset, batch_size=None)


def split_dataset(
    input_dir: pathlib.Path,
    out_dir: typing.Callable[[int], pathlib.Path],
    cluster: EricKMeans,
    bs: int,
    n_features: int,
    dtype: torch.dtype,
) -> None:
    """Take data from input_dir, classify it with cluster,
    and write it to the out_dir(i)"""

    for i in range(cluster.n_clusters):
        out_dir(i).mkdir(exist_ok=True, parents=True)
        # (out_dir(i) / "st").mkdir(exist_ok=True, parents=True)

    with contextlib.ExitStack() as stack:
        jsonl_files: list[typing.TextIO] = [
            stack.enter_context(open(out_dir(i) / "data.jsonl", "w"))
            for i in range(cluster.n_clusters)
        ]
        tensor_writers: list[SafetensorWriter] = [
            stack.enter_context(
                SafetensorWriter(
                    out_dir(i) / "embeddings.safetensors",
                    name="embeddings",
                    meta=TensorMetadata(
                        dtype=dtype,
                        # Infer embeddings shape by embedding an example text.
                        partial_shape=(n_features,),
                    ),
                )
            )
            for i in range(cluster.n_clusters)
        ]
        for i in range(cluster.n_clusters):
            tensor_writers[i].write_header()

        # Grab a batch.
        for batch in dataloader_for_dir(input_dir, bs=bs):
            # Classify each record in batch to a cluster ID.
            cluster_ids = cluster.predict(batch.embeddings).tolist()
            for text, embedding, meta, cluster_id in zip(
                batch.texts, batch.embeddings, batch.metas, cluster_ids
            ):
                # Write record to corresponding file for said cluster ID.
                jsonl_files[cluster_id].write(
                    json.dumps(
                        {
                            "text": text,
                            # 'embedding': embedding.tolist(),
                            "metadata": meta,
                        }
                    )
                    + "\n"
                )
                tensor_writers[cluster_id].write_append(embedding)


def embed_inputs(
    *,
    input_dir: pathlib.Path,
    out_dir: pathlib.Path,
    embed_fn: EmbedFn,
    input_count: int,
    bs: int,
    eric_timer: EricTimer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def iter_records() -> typing.Iterable[dict]:
        for filepath in input_dir.glob("*.jsonl"):
            with open(filepath) as f:
                while line := f.readline():
                    if line.strip():
                        record = json.loads(line)
                        yield record

    with contextlib.ExitStack() as exit_stack:
        f = exit_stack.enter_context(open(out_dir / "data.jsonl", "w"))
        tensor_writer = exit_stack.enter_context(
            SafetensorWriter(
                path=out_dir / "embeddings.safetensors",
                meta=TensorMetadata(
                    dtype=embed_fn(["x"]).dtype,
                    # Infer embeddings shape by embedding an example text.
                    partial_shape=(embed_fn(["x"]).shape[1],),
                ),
            )
        )
        tensor_writer.write_header()

        total_batches = math.ceil(input_count / bs)

        def batched(iterable, n):
            if n < 1:
                raise ValueError("Empty iterable")
            wrapped_it = iter(iterable)
            while batch := tuple(islice(wrapped_it, n)):
                yield batch

        batched_iter = batched(iter_records(), bs)

        pbar = tqdm(
            desc="Producing embeddings", total=total_batches
        )  # or omit total if unknown
        while True:
            with eric_timer.section(
                "file_operations", "Embeddings: get batched texts"
            ):
                try:
                    batch_records = next(batched_iter)
                except StopIteration:
                    break

            batch_texts = [r["text"] for r in batch_records]

            with eric_timer.section("embedding_model", "embed_fn"):
                embeddings = embed_fn(batch_texts)

            with eric_timer.section(
                "file_operations", "Embeddings: shard-write safetensors"
            ):
                tensor_writer.write_extend(embeddings)

            with eric_timer.section(
                "file_operations", "Generic: Embeddings: save embeddings"
            ):
                for record in batch_records:
                    f.write(
                        json.dumps(
                            {"text": record["text"], "metadata": record.get("metadata")}
                        )
                        + "\n"
                    )

            pbar.update(1)


T = typing.TypeVar("T")


def required(x: T | None) -> T:
    if x is None:
        raise ValueError("required() got None")
    else:
        return x
