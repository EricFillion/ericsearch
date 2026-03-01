import argparse
import json

from ericsearch import EricSearch, SearchTrainArgs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/output")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--branch", default="main")

    return p.parse_args()


def main(args: argparse.Namespace):

    eric_search = EricSearch()

    train_data = []

    for i in range(0, 100):
        train_data.append({"text": f"Hello world! This is case {i} out of 99.", "metadata": {"number": i}})

    with open(f"{args.raw_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for train_case in train_data:
            f.write(json.dumps(train_case) + "\n")

    train_args = SearchTrainArgs(leaf_size=4, out_dir=args.out_dir)

    eric_search.train(train_path=args.raw_dir, args=train_args)

    eric_search.push(repo_id=args.repo_id)





if __name__ == "__main__":
    main(get_args())
