import logging
import math
import re
import textwrap
from typing import List

from huggingface_hub import HfApi

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


def choose_k0_k1(
    input_count: int,
    k0: int | None = None,
    k1: int | None = None,
    *,
    leaf_size: int = 0,  # target items per final leaf
):
    if leaf_size < 1:
        leaf_size = 1024
        # leaf_size = choose_leaf_size(input_count)

    # 1) desired number of leaves
    M = max(1, math.ceil(input_count / leaf_size))

    if k0 is not None:
        if k0 < 1:
            k0 = None

    if k1 is not None:
        if k1 < 1:
            k1 = None

    # 2) solve for k0, k1
    if k0 is None and k1 is None:
        # balanced split: k0 * k1 ≈ M
        k0 = max(1, int(math.floor(math.sqrt(M))))
        k1 = int(math.ceil(M / k0))
    elif k0 is None and k1 is not None:
        k1 = max(1, int(k1))
        k0 = int(math.ceil(M / k1))
    elif k0 is not None and k1 is None:
        k0 = max(1, int(k0))
        k1 = int(math.ceil(M / k0))
    else:
        # both provided: ensure enough leaves
        # We should consider not auto adjusting k1.
        k0 = max(1, int(k0))
        k1 = max(1, int(k1))
        if k0 * k1 < M:
            # minimally bump k1 to cover M
            k1 = int(math.ceil(M / k0))

    print(f"k0={k0}, k1={k1}")
    return k0, k1


def make_readme(repo_id: str) -> str:
    readme_md = textwrap.dedent(f"""\
    ---
    tags:
    - ericsearch
    ---
    # {repo_id}

    ## Installation: 
    ```
    pip install ericsearch
    ```
    
    ## Usage 
    
    ```python
    from ericsearch import EricSearch
    
    eric_search = EricSearch(data_name="{repo_id}")

    result = eric_search('Hello world')
    
    print(result[0].text)
    ```
    
    See Eric Search's [GitHub](https://github.com/ericfillion/ericsearch) or [website](https://ericsearch.com/) for more information. 

    """)

    return readme_md


def ensure_readme(
    *, repo_name: str, logger: logging.Logger, api: HfApi, branch: str
) -> None:
    has_readme = api.file_exists(repo_name, "README.md", revision=branch, repo_type="dataset")

    if not has_readme:
        readme_text = make_readme(repo_name)
        logger.info("Pushing README...")

        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            revision=branch,
        )
