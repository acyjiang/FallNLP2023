import json
from pathlib import Path
import argparse

from noise_tools import noise_algorithm
from model.model import DenoiseModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", help="specify number of rows from dataset to use", type=int, default=100
    )
    return parser.parse_args()


def read_clean_dataset(n):
    dataset_path = Path(f"dataset{n}.json")

    if dataset_path.exists():
        with open(dataset_path, "r") as f:
            c4 = json.load(f)
            rows = c4["rows"]

            text = [row[0] for row in rows]
            print(text)


def generate_noisy_dataset(data):
    return [noise_algorithm(text) for text in data]


def main():
    args = get_args()

    clean_original_dataset = read_clean_dataset(args.n)

    noisy_dataset = generate_noisy_dataset(clean_original_dataset)

    model = DenoiseModel()


if __name__ == "__main__":
    main()
