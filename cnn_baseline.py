from pathlib import Path
import argparse

import datasets
from transformers import EncoderDecoderModel, BertTokenizer

from noise_tools import noise_algorithm

def add_noise(data):
    data["article"] = noise_algorithm(data["article"])
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", help="batch size", type=int, default=16
    )
    return parser.parse_args()

def main():
    args = get_args()
    batch_size = args.b

    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")
    test_data_noisy = test_data.map(add_noise)

    bert2bert = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda")
    tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

    def generate_summary(batch):
        # cut off at BERT max length 512
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred_summary"] = output_str

        return batch

    results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])
    results_noisy = test_data_noisy.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

    rouge = datasets.load_metric("rouge")
    print(rouge.compute(predictions=results["pred_summary"], references=results["highlights"], rouge_types=["rouge1", "rouge2", "rougeL"]))
    print(rouge.compute(predictions=results_noisy["pred_summary"], references=results["highlights"], rouge_types=["rouge1", "rouge2", "rougeL"]))

if __name__ == "__main__":
    main()