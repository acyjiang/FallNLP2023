from pathlib import Path
import argparse

import requests
import json
import tqdm


def get_dataset_url(offset):
    # URL for getting rows offset (inclusive) to offset + 100 (not inclusive) in the c4 dataset
    return f"https://datasets-server.huggingface.co/rows?dataset=c4&config=en&split=train&offset={offset}&length=100"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", help="specify number of rows from dataset", type=int, default=100
    )
    return parser.parse_args()


def main():
    args = get_args()
    dataset_path = Path(f"dataset{args.n}.json")

    if dataset_path.exists():
        print("Dataset already exists")
        return
    else:
        print("Dataset does not exist")

        with open(dataset_path, "w") as f:
            rows = []
            for i in tqdm.tqdm(range(0, args.n, 100)):
                dataset_url = get_dataset_url(i)
                response = requests.get(dataset_url)
                while response.status_code != 200:
                    print("Trying again...")
                    response = requests.get(dataset_url)
                c4 = response.json()

                for row in c4["rows"]:
                    text = row["row"]["text"]
                    rows.append(text)
                    json.dump({"clean": text}, f)
                    f.write("\n")

            print("Dataset created")


if __name__ == "__main__":
    main()
