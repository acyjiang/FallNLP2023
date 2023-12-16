import json
from pathlib import Path
import argparse


from noise_tools import noise_algorithm

import datasets

from transformers import EncoderDecoderModel, BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", help="batch size", type=int, default=16
    )
    parser.add_argument(
        "--path", help="path to checkpoint", type=str
    )
    return parser.parse_args()


def add_noise(data):
    data["noisy"] = noise_algorithm(data["clean"])
    return data


def main():
    args = get_args()
    batch_size = args.b

    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")
    test_data = test_data.map(add_noise)

    denoiser = EncoderDecoderModel.from_pretrained(args.path).to("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    summarizer = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda") 

    def generate_summary(batch):
        # cut off at BERT max length 512
        inputs = tokenizer(batch["noisy"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        cleaned_ids = denoiser.generate(input_ids, attention_mask=attention_mask)
        outputs = summarizer.generate(cleaned_ids, attention_mask=attention_mask)

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred_summary"] = output_str

        return batch
    
    results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["noisy"])

    rouge = datasets.load_metric("rouge")
    print(rouge.compute(predictions=results["pred_summary"], references=results["highlights"], rouge_types=["rouge1", "rouge2", "rougeL"]))    

if __name__ == "__main__":
    main()
          