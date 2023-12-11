import json
from pathlib import Path
import argparse


from noise_tools import noise_algorithm

from model.model import DenoiseModel
import datasets

from transformers import EncoderDecoderModel, BertTokenizerFast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", help="specify number of test rows from dataset to use", type=int, default=100
    )
    parser.add_argument(
        "--path", help="path to checkpoint", type=str
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

    clean_original_dataset = read_clean_dataset(n)

    test_data = clean_original_dataset.map(add_noise)

    finetuned_model = EncoderDecoderModel.from_pretrained(args.path).to("cuda")
    finetuned_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 

    def generate_clean(batch):
        # cut off at BERT max length 512
        inputs = finetuned_tokenizer(batch["noisy"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = finetuned_model.generate(input_ids, attention_mask=attention_mask)

        output_str = finetuned_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred_clean"] = output_str

        return batch
    
    test_data_small = test_data.select(range(args.n))

    result = test_data_small.map(generate_clean, batched=True, batch_size=4)

    for r in result['pred_clean']:
        print(r)
        

if __name__ == "__main__":
    main()
          