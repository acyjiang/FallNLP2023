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
            dataset = datasets.load_dataset(
                "json", data_files=f"dataset{n}.json", split="train"
            )
        return dataset
    else:
        print("Dataset does not exist")
        return


def add_noise(data):
    data["noisy"] = noise_algorithm(data["clean"])
    return data


def get_data_splits(dataset):
    split = 0.15

    train_test_split = dataset.train_test_split(test_size=split)

    train_val_data = train_test_split["train"]

    train_val_split = train_val_data.train_test_split(test_size=split / (1 - split))

    train_data = train_val_split["train"]

    val_data = train_val_split["test"]

    test_data = train_test_split["test"]

    return train_data, val_data, test_data


def main():
    args = get_args()

    clean_original_dataset = read_clean_dataset(args.n)

    full_dataset = clean_original_dataset.map(add_noise)

    train_data, val_data, test_data = get_data_splits(full_dataset)


    model = DenoiseModel()

    model.train(train_data, val_data)

    # # Results
    # denoised_data = [model(text) for text in noisy_data]

    dummy_bert2bert = model.from_pretrained("./checkpoint-20")


if __name__ == "__main__":
    main()
