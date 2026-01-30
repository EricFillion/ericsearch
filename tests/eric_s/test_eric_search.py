import contextlib
import pathlib
from unittest.mock import Mock

import pytest
import torch

from tests.get_test_models import (
    CROSS_ENCODER_PATH,
    EMBEDDING_MODEL_PATH,
)
from ericsearch import EricRanker, EricSearch, SearchTrainArgs, SearchCallArgs
import ericsearch.eric_search as eric_search_module


def pad_or_trim(items: list[float], target_len: int):
    items = items[:target_len]
    items += [0] * (target_len - len(items))
    return items


vector_data = """
{ "text": "Ottawa is the capital of Canada", "metadata": {"id": "a" } }
{ "text": "Kingston was once the capital of Canada", "metadata": {"id": "b" }  }
{ "text": "Toronto is the largest city in Canada", "metadata": {"id": "c" }  }
""".strip()


class MockEmbeddingsModel:
    """Very naive embeddings model that runs super fast."""

    invert: bool

    def __init__(self) -> None:
        self.invert = False

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Literally just number characters. Works in this tiny example.
        emb = torch.tensor(
            [pad_or_trim([float(ord(char)) for char in text], 10) for text in texts]
        )
        if self.invert:
            emb *= -1
        return emb

    def metadata(self):
        return "metadata"


def test_train_eric_search(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()
    input_str_path = str((tmp_path / "inputs"))
    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

    eric_search = EricSearch(model_name=EMBEDDING_MODEL_PATH, eric_ranker=eric_ranker)
    eric_search.train(
        train_path=input_str_path,
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
    )

    # train it twice
    eric_search.train(
        input_str_path,
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_1")),
    )

    first_prediction = eric_search("ottawa")
    print(f"{first_prediction=}")

    eric_search = EricSearch(
        data_name=str(tmp_path / "output_1"),
        model_name=EMBEDDING_MODEL_PATH,
        eric_ranker=eric_ranker,
    )

    second_prediction = eric_search("ottawa")
    print(f"{second_prediction=}")

    assert first_prediction == second_prediction


def test_custom_embedding_model(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

    eric_search = EricSearch(model_name=EMBEDDING_MODEL_PATH, eric_ranker=eric_ranker)

    eric_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    first_prediction = eric_search("ottawa")
    print(f"{first_prediction=}")

    eric_search = EricSearch(
        data_name=str(tmp_path / "output"),
        model_name=EMBEDDING_MODEL_PATH,
        eric_ranker=eric_ranker,
    )

    second_prediction = eric_search("ottawa",
                                    args=SearchCallArgs(bs=2, limit=1))
    print(f"{second_prediction=}")

    assert first_prediction == second_prediction


def test_check_other_clusters(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

    eric_search = EricSearch(
        # Use a non-default model such that this test
        # asserts the same model is used after the load.
        model_name=EMBEDDING_MODEL_PATH,
        eric_ranker=eric_ranker,
    )
    # Tell the fine-grained information extractor
    # not to weigh the initial score at all,
    # as we are intentionally screwing it up.
    eric_search.eric_ranker.ignore_original_score = True

    eric_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    first_prediction = eric_search("ottawa")
    print(f"{first_prediction=}")

    second_prediction = eric_search("ottawa")
    print(f"{second_prediction=}")

    assert first_prediction[0].text == second_prediction[0].text


def test_metadata(tmp_path: pathlib.Path):
    """Assert that metadata from training appears in inference query."""
    (tmp_path / "inputs").mkdir()
    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_ranker = EricRanker(model_name=CROSS_ENCODER_PATH)

    eric_search = EricSearch(model_name=EMBEDDING_MODEL_PATH, eric_ranker=eric_ranker)

    eric_search.eric_ranker.ignore_original_score = True
    eric_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(
            out_dir=str(tmp_path / "output_0"),
        )
    )

    prediction = eric_search("ottawa")
    assert prediction[0].text == "Ottawa is the capital of Canada"
    assert prediction[0].metadata == {"id": "a"}



def test_push_smoke(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_search = EricSearch()
    eric_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:
        eric_search.push(repo_id="dontcare")


@contextlib.contextmanager
def mock_hf_api():
    original = eric_search_module.HfApi
    try:
        original_sleep = eric_search_module.sleep
        eric_search_module.sleep = lambda *a, **k: None

        mock = Mock()
        mock.list_repo_files = lambda *a, **k: []
        eric_search_module.HfApi = lambda: mock

        yield mock
    finally:
        eric_search_module.HfApi = original
        eric_search_module.sleep = original_sleep


def test_push_exceed_retries(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_search = EricSearch()
    eric_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:

        def fn(*a, **k):
            raise Exception("fake error")

        api.create_commit = fn
        with pytest.raises(RuntimeError):
            eric_search.push(repo_id="dontcare")


def test_push_within_retry(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    eric_search = EricSearch()
    eric_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:

        def fn(*a, commit_message: str, **k):
            if "retry" not in commit_message:
                raise Exception("fake error")

        api.create_commit = fn

        # This should succeed since we only err before retry
        eric_search.push(repo_id="dontcare")
