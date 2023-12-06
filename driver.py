import json
from pathlib import Path
import argparse

from noise_tools import noise_algorithm

from model.model import DenoiseModel
import datasets


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
            dataset = datasets.load_dataset("json", data_files=f"dataset{n}.json")
        return dataset
    else:
        print("Dataset does not exist")
        return


def add_noise(data):
    data["noisy"] = noise_algorithm(data["clean"])
    return data


def main():
    args = get_args()

    clean_original_dataset = read_clean_dataset(args.n)

    full_dataset = clean_original_dataset.map(add_noise)

    print(clean_original_dataset)
    print(full_dataset)


    # model.train(dataset)

    # # Results
    # denoised_data = [model(text) for text in noisy_data]


if __name__ == "__main__":
    main()
