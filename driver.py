import json
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", help="specify number of rows from dataset to use", type=int, default=100
    )
    return parser.parse_args()


def main():
    args = get_args()
    dataset_path = Path(f"dataset{args.n}.json")

    if dataset_path.exists():
        with open(dataset_path, "r") as f:
            c4 = json.load(f)
            rows = c4["rows"]

            text = [row[0] for row in rows]
            print(text)


if __name__ == "__main__":
    main()
