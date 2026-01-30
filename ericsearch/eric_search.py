from dataclasses import dataclass
import math
import os
import pathlib
import time
import typing

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils.tqdm import disable_progress_bars, enable_progress_bars
from torch import device

from ericsearch.db.ericdata_dir import EricSearchDirStateReader
from ericsearch.train import EricKMeans
from ericsearch.eric_ranker import EricRanker, RankerCallArgs
from ericsearch.utils.misc import choose_k0_k1, ensure_readme
from ericsearch.utils import (
    EricVector,
    EricTimer,
    es_get_device,
    es_get_logger,
    load_from_repo_or_path,
    EmbeddingsModel,
    EricDocument,
    SearchResult
)
from ericsearch.train import train_eric_search

DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def resolve_embeddings_model(
    embeddings_model: str | None,
    device: device | None = None,
    bs: int | None = None,
) -> EmbeddingsModel:
    if embeddings_model is None:
        embeddings_model = DEFAULT_EMBEDDINGS_MODEL

    if bs is None:
        bs = 16

    if device is None:
        device = es_get_device()

    return EricVector(
        model_name_or_path=embeddings_model, bs=bs, d=device
    )

@dataclass(kw_only=True)
class SearchTrainArgs:
    out_dir: str = "eric_search/"
    leaf_size: int = 0
    bs: int = 32

@dataclass(kw_only=True)
class SearchCallArgs:
    limit: int = 1
    leaf_count: int = 32
    ranker_count: int = 4
    bs: int = 32


class EricSearch:
    def __init__(
        self,
        *
        ,
        data_name: str | None = None,
        model_name: str | None = None,
        eric_ranker: EricRanker | None = None,
    ):
        if eric_ranker is not None:
            self.eric_ranker = eric_ranker
        else:
            self.eric_ranker = EricRanker()

        self.device = es_get_device()
        self.logger = es_get_logger()

        # If state dir is supplied, load directly.
        if data_name:
            fields = load_ericsearch_fields(
                data_name,
                d=self.device,
                eric_timer=None,
            )

            for key, value in fields.items():
                if key == "embeddings_model":
                    if value is not None:
                        # allow user to use their own model_name rather than the default
                        em = value.model_name_or_path
                        if model_name != value.model_name_or_path:
                            if model_name is not None:
                                self.logger.info(
                                    f"The provided model_name ({model_name}) does not match the EricData's embeddings model {value.model_name_or_path}. This is okay if they're both the same model, just saved in different locations. model_name takes priority. "
                                )
                                em = model_name

                        self.embeddings_model = resolve_embeddings_model(em)
                        continue

                setattr(self, key, value)
            return

        # Otherwise, setup config for training.

        # These fields define the model's config.
        # They are always meaningful.

        self.embeddings_model = resolve_embeddings_model(model_name)

        # These fields are used for inference.
        # They are only meaningful post-train.
        self.ivf0_cluster = None
        self.ivf1_clusters = []
        self.state_reader = None

    def train(self, train_path: str, args: SearchTrainArgs = SearchTrainArgs()) -> None:
        # Train the object, saving output to disk as we go.
        input_count = 0
        input_dir_path = pathlib.Path(train_path)
        out_dir_path = pathlib.Path(args.out_dir)
        timer_path = out_dir_path.joinpath("time")
        eric_timer = EricTimer(str(timer_path))
        eric_timer.report()
        with eric_timer.section("file_operations", "Generic: Iterate over input_dir"):
            for filepath in input_dir_path.glob("*.jsonl"):
                with open(filepath) as f:
                    while line := f.readline():
                        if line.strip():
                            input_count += 1

        # we can consider making k0 and k1 configurable in the future.
        # for now the users can just adjust leaf size.
        self.k0, self.k1 = choose_k0_k1(
            k0=0, k1=0, input_count=input_count, leaf_size=args.leaf_size
        )

        if self.ivf0_cluster is None:
            self.ivf0_cluster = EricKMeans(n_clusters=self.k0, eric_timer=eric_timer)

        with eric_timer.section("eric_k_means", "Create IVF-1 clusters"):
            if not self.ivf1_clusters:
                self.ivf1_clusters = [
                    EricKMeans(n_clusters=self.k1, eric_timer=eric_timer)
                    for _ in range(self.k0)
                ]
        eric_timer.report()

        # Train and save results to disk.
        train_eric_search(
            k0=self.k0,
            k1=self.k1,
            input_count=input_count,
            input_dir=input_dir_path,
            out_dir=out_dir_path,
            embeddings_model=self.embeddings_model,
            bs=args.bs,
            bs_kmeans=args.bs,
            device=self.device,
            logger=self.logger,
            eric_timer=eric_timer,
        )

        with eric_timer.section("file_operations", "Generic: Load final clusters"):
            # Load trained model back into fields of `self`.
            self.state_reader = EricSearchDirStateReader(args.out_dir, d=self.device)
            centroids = self.state_reader.load_centroids()
            self.ivf0_cluster.centroids = centroids.ivf0_centroids
            for ivf1_cluster, ivf1_centroids in zip(
                self.ivf1_clusters, centroids.ivf1_centroids
            ):
                ivf1_cluster.centroids = ivf1_centroids
        eric_timer.report()

    @torch.no_grad()
    def __call__(self, text: str, args: SearchCallArgs = SearchCallArgs()) -> list[SearchResult]:
        """Run inference on a text."""

        c0_k = max(1, int(math.floor(math.sqrt(args.leaf_count))))
        c1_k = int(math.ceil(args.leaf_count / c0_k))

        if not self.state_reader:
            raise ValueError(
                "state_reader is unset. Call train() or load from existing dir?"
            )

        if not self.ivf0_cluster:
            raise ValueError("ivf0_cluster is unset")

        # Encode query and pick IVF0 (top-level) cluster
        vector = self.embeddings_model.encode([text])  # [1, D]
        vector = vector.to(device=self.device, dtype=torch.float32)

        # Look into a few top-level clusters.
        c0s = self.ivf0_cluster.predict_multi(vector, k=c0_k)[0].tolist()

        # For each selected top level cluster,
        # look into a few subclusters.
        c0_c1s = [
            self.ivf1_clusters[c0].predict_multi(vector, k=c1_k)[0].tolist()
            for c0 in c0s
        ]

        # Grab all embeddings and texts such that
        # we can compare every selected cluster in one big pass.
        embs: list[torch.Tensor] = []
        texts: list[str] = []
        metas: list[typing.Any] = []
        for i, c0 in enumerate(c0s):
            for c1 in c0_c1s[i]:
                cluster = self.state_reader.load_cluster(c0, c1)
                emb = cluster.embeddings.to(device=self.device, dtype=torch.float32)
                embs.append(emb)
                texts.extend(cluster.texts)
                metas.extend(cluster.metas)

        emb = torch.concat(embs)

        sims = (vector @ emb.T).squeeze(0)  # [M]
        top_scores, top_indices = torch.topk(sims, min(args.ranker_count, sims.numel()))

        doc_search_results = [
            EricDocument(text=texts[i], score=score, metadata=metas[i])
            for i, score in zip(top_indices, top_scores.tolist())
        ]

        ranker_args = RankerCallArgs(bs=args.bs, limit=args.limit)

        results = self.eric_ranker(
            text=text, docs=doc_search_results, args=ranker_args
        )

        output = []
        for result in results:
            output.append(
                SearchResult(
                    text=result.text,
                    score=result.score,
                    metadata=result.metadata,
                    best_sentence=result.best_sentence,
                )
            )

        return output

    def push(
        self,
        repo_id: str,
        *,
        bs: int = 4,
        branch: str = "main",
        overwrite: bool = False,
        private: bool = True,
    ) -> None:
        api = HfApi()

        if not self.state_reader:
            raise ValueError("Must have local state to push")

        # Ensure repo exists
        api.create_repo(
            repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset"
        )

        # Ensure branch exists
        api.create_branch(
            repo_id=repo_id,
            repo_type="dataset",
            branch=branch,
            revision="main",
            exist_ok=True,
        )

        hf_paths = set(
            api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=branch)
        )

        root = self.state_reader.root

        def _chunked(iterable, size):
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) == size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def iter_operations():
            skip_dirs = {
                ".ipynb_checkpoints",
                "time",
                "artifacts",
                "tmp",
                ".cache",
                ".git",
                "__pycache__",
            }

            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]

                for filename in filenames:
                    local_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(local_path, root)

                    path_in_repo = f"eric_search/{rel_path}".replace("\\", "/")

                    if overwrite or (path_in_repo not in hf_paths):
                        self.logger.info("Adding %s", local_path)
                        yield CommitOperationAdd(
                            path_in_repo=path_in_repo, path_or_fileobj=local_path
                        )
                    else:
                        self.logger.info("Skipping %s", local_path)

        for i, ops_batch in enumerate(_chunked(iter_operations(), bs), start=1):
            self.logger.info("Committing...")
            success = False
            try:
                # first attempt
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=ops_batch,
                    commit_message=f"batch {i}",
                    num_threads=4,
                    revision=branch,
                )
                success = True
            except Exception as e:
                self.logger.warning(
                    f"Failed to push commit. Sleeping for two minutes. {e}"
                )
                sleep(120.0)

            if not success:
                # second attempt
                try:
                    self.logger.info("Trying again")
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=ops_batch,
                        commit_message=f"batch {i} retry",
                        num_threads=2,
                        revision=branch,
                    )
                    success = True
                except Exception as e:
                    self.logger.warning(
                        f"Failed to push commit. Sleeping for five minutes. {e}"
                    )
                    sleep(300.0)

            if not success:
                # third attempt
                try:
                    self.logger.info("Trying again (final attempt)")
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=ops_batch,
                        commit_message=f"batch {i} retry 2",
                        num_threads=1,
                        revision=branch,
                    )
                    success = True
                except Exception as e:
                    raise RuntimeError(
                        "Failed to push commit after two re-tries. Ending"
                    ) from e

            self.logger.info("Success. Sleeping for 0.1 seconds...\n\n")
            sleep(0.1)

        self.logger.info("\n\n\nUploading README")

        try:
            ensure_readme(
                repo_name=repo_id, logger=self.logger, api=api, branch=branch
            )
        except Exception as e:
            self.logger.warning(
                f"Error ensuring README exists, but data uploaded OK: {e}"
            )

        self.logger.info("DONE")


def load_ericsearch_fields(
    dir_path: str,
    d: torch.device,
    eric_timer=None,
):
    # Resolve possibly remote path to local dir path.
    dir = load_from_repo_or_path(dir_path, path_in_repo="eric_search")

    disable_progress_bars()  # globally

    enable_progress_bars()

    if not dir:
        raise ValueError("dir is None")

    # Init state reader and eagerly load centroids.
    state_reader = EricSearchDirStateReader(dir, d=d)

    embeddings_model = state_reader.load_embeddings_model()

    centroids = state_reader.load_centroids()
    k0 = len(centroids.ivf1_centroids)
    k1 = centroids.ivf1_centroids[0].shape[0]

    ivf0_cluster = EricKMeans(n_clusters=k0, eric_timer=eric_timer)
    ivf1_clusters = [
        EricKMeans(n_clusters=k1, eric_timer=eric_timer) for _ in range(k0)
    ]

    ivf0_cluster.centroids = centroids.ivf0_centroids
    for i, ivf1_cluster in enumerate(ivf1_clusters):
        ivf1_cluster.centroids = centroids.ivf1_centroids[i]

    return dict(
        embeddings_model=embeddings_model,
        state_reader=state_reader,
        ivf0_cluster=ivf0_cluster,
        ivf1_clusters=ivf1_clusters,
        k0=k0,
        k1=k1,
    )


# This wrapper is here for mocking during testing.
def sleep(seconds: float) -> None:
    time.sleep(seconds)
