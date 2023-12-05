import requests
import json
from urllib.parse import quote_plus
import io
from bs4 import BeautifulSoup
from pathlib import Path

dataset_url = "https://datasets-server.huggingface.co/rows?dataset=c4&config=en&split=train&offset=0&length=100"


def main():
    dataset_path = Path("dataset.json")
    if dataset_path.exists():
        print("Dataset already exists")
        return
    else:
        print("Dataset does not exist")
        response = requests.get(dataset_url)
        c4 = response.json()

        dataset_path.touch()
        with open(dataset_path, "w") as f:
            json.dump(c4, f)


if __name__ == "__main__":
    main()
